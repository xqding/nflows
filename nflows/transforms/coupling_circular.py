"""Implementations of various coupling layers."""
import warnings

import numpy as np
import torch

from nflows.transforms import splines
from nflows.transforms.base import Transform
from nflows.transforms.nonlinearities import (
    PiecewiseCubicCDF,
    PiecewiseLinearCDF,
    PiecewiseQuadraticCDF,
    PiecewiseRationalQuadraticCDF,
)
from nflows.utils import torchutils
from nflows.transforms.coupling import (
    CouplingTransform,
    PiecewiseCouplingTransform
)

class PiecewiseCircularRationalQuadraticCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        ## constraint the derivatives at the two end points are the same
        unnormalized_derivatives=torch.cat([unnormalized_derivatives, unnormalized_derivatives[..., 0][..., None]], dim = -1)
        
        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {"left": -np.pi, "right": np.pi, "bottom": -np.pi, "top": np.pi}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )
