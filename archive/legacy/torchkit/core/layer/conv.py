#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution Layers.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from torchkit.core.utils import to_2tuple
from .builder import CONV_LAYERS
from .padding import get_padding_value
from .padding import pad_same

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None
    
logger = logging.getLogger()


# MARK: - Register

CONV_LAYERS.register(name="conv",    module=nn.Conv2d)
CONV_LAYERS.register(name="conv_1d", module=nn.Conv1d)
CONV_LAYERS.register(name="conv_2d", module=nn.Conv2d)
CONV_LAYERS.register(name="conv_3d", module=nn.Conv3d)


# MARK: - CondConv2d

def get_condconv_initializer(initializer, num_experts: int, expert_shape):
    def condconv_initializer(weight: torch.Tensor):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2 or
            weight.shape[0] != num_experts or
            weight.shape[1] != num_params
        ):
            raise (ValueError("CondConv variables must have shape "
                              "[num_experts, num_params]"))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


@CONV_LAYERS.register(name="cond_conv_2d")
class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution. Inspired by:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (3, 3),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = "",
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : Optional[bool]           = False,
        num_experts : int                      = 4
    ):
        super(CondConv2d, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = to_2tuple(kernel_size)
        self.stride       = to_2tuple(stride)

        padding_val, is_padding_dynamic = get_padding_value(
			padding, kernel_size, stride=stride, dilation=dilation
		)
        # if in forward to work with torchscript
        self.dynamic_padding = is_padding_dynamic
        self.padding         = to_2tuple(padding_val)
        self.dilation        = to_2tuple(dilation)
        self.groups          = groups
        self.num_experts     = num_experts

        self.weight_shape = (
			(self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(
			torch.Tensor(self.num_experts, weight_num_param)
		)

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(
				torch.Tensor(self.num_experts, self.out_channels)
			)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
			self.num_experts, self.weight_shape
        )
        init_weight(self.weight)
        if self.bias is not None:
            fan_in    = np.prod(self.weight_shape[1:])
            bound     = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts,
				self.bias_shape
            )
            init_bias(self.bias)

    # MARK: Forward Pass

    def forward(
		self, x: torch.Tensor, routing_weights: torch.Tensor
	) -> torch.Tensor:
        b, c, h, w = x.shape
        weight     = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (
			(b * self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight = weight.view(new_weight_shape)
        bias   = None

        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(b * self.out_channels)
        # Move batch elements with channels so each batch element can be
		# efficiently convolved with separate kernel
        x = x.view(1, b * c, h, w)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        out = out.permute([1, 0, 2, 3]).view(b, self.out_channels,
											 out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


CondConv = CondConv2d
CONV_LAYERS.register(name="cond_conv", module=CondConv)


# MARK: - Conv2dTF

@CONV_LAYERS.register(name="conv_2d_tf")
class Conv2dTF(nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride `1`, this will
    ensure that output image size is same as input. For stride of 2, output
    dimensions will be half, for example.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel
        stride (Size2T):
            Stride of the convolution. Default: `1`.
        padding (Size2T):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (str, Size2T, optional):
            Spacing between kernel elements. Default: `1`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.
        bias (bool):
            If `True`, adds a learnable bias to the output. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T                   = 1,
        padding     : Union[str, Size2T, None] = 0,
        dilation    : Size2T                   = 1,
        groups      : int                      = 1,
        bias        : bool                     = True,
        *args, **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, *args, **kwargs
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor):
                The input tensor.

        Returns:
            y_hat (torch.Tensor):
                The output tensor.
        """
        img_h, img_w       = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h    = max((output_h - 1) * self.stride[0] + (kernel_h - 1) *
                       self.dilation[0] + 1 - img_h, 0)
        pad_w    = max((output_w - 1) * self.stride[1] + (kernel_w - 1) *
                       self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        y_hat = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                         self.dilation, self.groups)
        return y_hat


ConvTF = Conv2dTF
CONV_LAYERS.register(name="conv_tf", module=ConvTF)


# MARK: - Conv2dSame

def conv2d_same(
    x       : torch.Tensor,
    weight  : torch.Tensor,
    bias    : Optional[torch.Tensor]   = None,
    stride  : Size2T                   = (1, 1),
    padding : Union[Size4T, str, None] = 0,
    dilation: Size2T                   = (1, 1),
    groups  : int                      = 1,
    **_
):
    """Functional interface for Same Padding Convolution 2D.

    Args:
        x (torch.Tensor):
            The input tensor.
        weight (torch.Tensor):
            The weight.
        bias (torch.Tensor, optional):
            The bias value.
        stride (Size2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Size4T, str, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Size2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.

    Returns:
        x (torch.Tensor):
            The output tensor.
    """
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


@CONV_LAYERS.register(name="conv_2d_same")
class Conv2dSame(nn.Conv2d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel. Default: `(1, 1)`.
        stride (Size2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Size4T, str, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Size2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        groups (int):
            Default: `1`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = 0,
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_same(x, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)


ConvSame = Conv2dSame
CONV_LAYERS.register(name="conv_same", module=ConvSame)


# MARK: - MixedConv2d

def _split_channels(num_channels: int, num_groups: int):
    split     = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


@CONV_LAYERS.register(name="mixed_conv_2d")
class MixedConv2d(nn.ModuleDict):
    """Mixed Convolution from the paper `MixConv: Mixed Depthwise
    Convolutional Kernels` (https://arxiv.org/abs/1907.09595)

    Based on MDConv and GroupedConv in MixNet implementation:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (3, 3),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = "",
        dilation    : Size2T                   = (1, 1),
        depthwise   : bool                     = False,
        **kwargs
    ):
        super().__init__()
        kernel_size       = kernel_size
        stride            = to_2tuple(stride)
        dilation          = to_2tuple(dilation)
        num_groups        = len(kernel_size)
        in_splits         = _split_channels(in_channels, num_groups)
        out_splits        = _split_channels(out_channels, num_groups)
        self.in_channels  = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = in_ch if depthwise else 1
            # Use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride, padding=padding,
                    dilation=dilation, groups=conv_groups, **kwargs
                )
            )
        self.splits = in_splits

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.splits, 1)
        x_out   = [c(x_split[i]) for i, c in enumerate(self.values())]
        x       = torch.cat(x_out, 1)
        return x


MixedConv = MixedConv2d
CONV_LAYERS.register(name="mixed_conv", module=MixedConv)


# MARK: - Builder

def create_conv2d_pad(
    in_channels: int, out_channels: int, kernel_size: Size2T, **kwargs
) -> nn.Conv2d:
    """Create 2D Convolution layer with padding."""
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)

    if is_dynamic:
        return Conv2dSame(in_channels, out_channels, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size,
                         padding=padding, **kwargs)


def create_conv2d(
    in_channels: int, out_channels: int, kernel_size: Size2T, **kwargs
):
    """Select a 2d convolution implementation based on arguments. Creates and
    returns one of `torch.nn.Conv2d`, `Conv2dSame`, `MixedConv2d`, or
    `CondConv2d`. Used extensively by EfficientNet, MobileNetv3 and related
    networks.
    """
    if isinstance(kernel_size, list):
        # MixNet + CondConv combo not supported currently
        assert "num_experts" not in kwargs
        # MixedConv groups are defined by kernel list
        assert "groups"      not in kwargs
        # We're going to use only lists for defining the MixedConv2d kernel
        # groups, ints, tuples, other iterables will continue to pass to
        # normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop("depthwise", False)
        # for DW out_channels must be multiple of in_channels as must have
        # out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            m = CondConv2d(
				in_channels, out_channels, kernel_size, groups=groups,
				**kwargs
			)
        else:
            m = create_conv2d_pad(
				in_channels, out_channels, kernel_size, groups=groups, **kwargs
			)
    return m
