#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

logger = logging.getLogger()


# MARK: - Functional API

def constant_init(
    module: nn.Module,
    val   : Union[int, float],
    bias  : Union[int, float] = 0.0
):
    """Initialize module with constant value and bias.

    Args:
        module (nn.Module):
            The module will be initialized.
        val (int, float):
            The constant value.
        bias (int, float):
            The bias. Default: `0.0`.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(tensor=module.weight, val=val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)


def xavier_init(
    module      : nn.Module,
    gain        : Union[int, float] = 1.0,
    bias        : Union[int, float] = 0.0,
    distribution: str = "normal"
):
    """Xavier initialization. Initialize module with values according to the
    method described in `Understanding the difficulty of training deep
    feedforward neural networks` - Glorot, X. & Bengio, Y. (2010), using either
    a normal or uniform distribution.

    Args:
        module (nn.Module):
            The module will be initialized.
        gain (int, float):
            An optional scaling factor. Default: `1`.
        bias (int, float):
            The value to fill the bias. Default: `0`.
        distribution (str):
            One of: [`normal`, `uniform`]. Default: `normal`.
    """
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(tensor=module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(tensor=module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)


def normal_init(
    module: nn.Module,
    mean  : Union[int, float] = 0.0,
    std   : Union[int, float] = 1.0,
    bias  : Union[int, float] = 0.0
):
    """Fills the module with values drawn from the normal distribution.

    Args:
        module (nn.Module):
            The module will be initialized.
        mean (int, float):
            The mean of the normal distribution. Default: `0.0`.
        std (int, float):
            The standard deviation of the normal distribution. Default: `1.0`.
        bias (int, float):
            Default: `0.0`.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(tensor=module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)


def trunc_normal_init(
    module: nn.Module,
    mean  : Union[int, float] = 0.0,
    std   : Union[int, float] = 1.0,
    a     : Union[int, float] = -2.0,
    b     : Union[int, float] = 2.0,
    bias  : Union[int, float] = 0.0
) -> None:
    """

    Args:
        module:
            The module will be initialized.
        mean (float):
            The mean of the normal distribution. Default: `0`.
        std (float):
            The standard deviation of the normal distribution. Default: `1`.
        a (float):
            The minimum cutoff value. Default: `-2`.
        b ( float):
            The maximum cutoff value. Default: `2`.
        bias (float):
            The value to fill the bias. Default: `0`.
    """
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


def uniform_init(
    module: nn.Module,
    a   : Union[int, float] = 0.0,
    b   : Union[int, float] = 1.0,
    bias: Union[int, float] = 0.0
):
    """

    Args:
        module:
            The module will be initialized.
        a (float):
            The minimum cutoff value. Default: `-2`.
        b ( float):
            The maximum cutoff value. Default: `2`.
        bias (float):
            The value to fill the bias. Default: `0`.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.uniform_(tensor=module.weight, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)


def kaiming_init(
    module      : nn.Module,
    a           : Union[int, float] = 0.0,
    mode        : str               = "fan_out",
    nonlinearity: str               = "relu",
    bias        : Union[int, float] = 0.0,
    distribution: str               = "normal"
):
    """Initialize module with values according to the method described in
    `Delving deep into rectifiers: Surpassing human-level performance on
    ImageNet classification` - He, K. et al. (2015), using either a normal or
    an uniform distribution.

    Args:
        module (nn.Module):
            The module will be initialized.
        a (int, float):
            The negative slope of the rectifier used after this layer (only
            used with `leaky_relu`). Default: `0`.
        mode (str):
            One of: [`fan_in`, `fan_out`]. Choosing `fan_in` preserves the
            magnitude of the variance of the weights in the forward pass.
            Choosing `fan_out` preserves the magnitudes in the backwards pass.
            Defaults to `fan_out`.
        nonlinearity (str):
            The non-linear function (`nn.functional` name), recommended to use
            only with `relu` or `leaky_relu`.
            Default: `relu`.
        bias (int, float):
            The value to fill the bias. Default: `0`.
        distribution (str):
            One of: [`normal`, `uniform`]. Default: `normal`.
    """
    assert distribution in ["normal", "uniform"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(tensor=module.weight, a=a, mode=mode,
                                     nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(tensor=module.weight, a=a, mode=mode,
                                    nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)


def caffe2_xavier_init(module: nn.Module, bias: Union[int, float] = 0.0):
    """`XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch.
    Acknowledgment to FAIR's internal code.

    Args:
        module (nn.Module):
            The module to initialize.
        bias (int, float):
            Default: `0.0`.
    """
    kaiming_init(
        module       = module,
        a            = 1,
        mode         = "fan_in",
        nonlinearity = "leaky_relu",
        bias         = bias,
        distribution = "uniform"
    )


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability value.
    """
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


# noinspection PyProtectedMember
def update_init_info(module: nn.Module, init_info: str):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (nn.Module):
            The module of PyTorch with a user-defined attribute
            `_params_init_info` which records the initialization information.
        init_info (str):
            The string that describes the initialization.
    """
    assert hasattr(module, "_params_init_info"), \
        f"Can not find `_params_init_info` in {module}."

    for name, param in module.named_parameters():
        assert param in module._params_init_info, (
            f"Find a new :obj:`Parameter` named `{name}` during executing the "
            f"`init_weights` of `{module.__class__.__name__}`. "
            f"Please do not add or replace parameters during executing the "
            f"`init_weights`."
        )

        # NOTE: The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:
            module._params_init_info[param]["init_info"] = init_info
            module._params_init_info[param]["tmp_mean_value"] = mean_value


def _get_bases_name(m):
    return [b.__name__ for b in m.__class__.__bases__]


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    # Cut & paste from PyTorch official master until it's in a few official
    # releases - RW Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
			"mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
			stacklevel=2
		)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
	tensor: torch.Tensor,
	mean  : float = 0.0,
	std   : float = 1.0,
	a     : float = -2.0,
	b     : float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated  normal
    distribution. The values are effectively drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)` with values
    outside :math:`[a, b]` redrawn until they are within the bounds. The
    method used for generating the random values works best when :math:`a
    \leq \text{mean} \leq b`.

    Args:
        tensor (torch.Tensor):
        	An n-dimensional `torch.Tensor`.
        mean (float):
        	The mean of the normal distribution.
        std (flaot):
        	The standard deviation of the normal distribution.
        a (float):
        	The minimum cutoff value.
        b (float):
        	The maximum cutoff value.

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(
    tensor      : torch.Tensor,
    scale       : float        = 1.0,
    mode        : str          = "fan_in",
    distribution: str          = "normal"
):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor: torch.Tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# MARK: - Initializing Helper Functions

def trunc_normal_(
    tensor: torch.Tensor,
    mean  : float = 0.0,
    std   : float = 1.0,
    a     : float = -2.0,
    b     : float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated normal
    distribution. The values are effectively drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)` with values
    outside :math:`[a, b]` redrawn until they are within the bounds. The
    method used for generating the random values works best when :math:`a
    \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (torch.Tensor):
            An n-dimensional `torch.Tensor`.
        mean (float):
            The mean of the normal distribution. Default: `0.0`.
        std (float):
            The standard deviation of the normal distribution. Default: `1.0`.
        a (float):
            The minimum cutoff value. Default: `-2.0`.
        b (float):
            The maximum cutoff value. Default: `2.0`.
    """
    return _no_grad_trunc_normal_(tensor=tensor, mean=mean, std=std, a=a, b=b)


def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean  : float,
    std   : float,
    a     : float,
    b     : float
) -> torch.Tensor:
    """Method based on: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (torch.Tensor):
            An n-dimensional `torch.Tensor`.
        mean (float):
            The mean of the normal distribution. Default: `0.0`.
        std (float):
            The standard deviation of the normal distribution. Default: `1.0`.
        a (float):
            The minimum cutoff value. Default: `-2.0`.
        b (float):
            The maximum cutoff value. Default: `2.0`.

    Returns:

    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
