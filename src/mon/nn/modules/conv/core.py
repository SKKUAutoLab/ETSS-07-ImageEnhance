#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements convolutional layers."""

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv2dNormAct",
    "Conv2dNormActivation",
    "Conv2dSame",
    "Conv3d",
    "Conv3dNormAct",
    "Conv3dNormActivation",
    "ConvNormAct",
    "ConvNormActivation",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "conv2d_same",
]

from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t, _size_any_t
from torch.nn.modules.conv import (
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    LazyConv1d, LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d,
    LazyConvTranspose3d,
)
from torchvision.ops.misc import (
    Conv2dNormActivation, Conv3dNormActivation, ConvNormActivation,
)

from mon.nn.modules import padding as pad


# ----- Conv2dSame -----
def conv2d_same(
    input   : torch.Tensor,
    weight  : torch.Tensor,
    bias    : torch.Tensor = None,
    stride  : _size_any_t  = 1,
    padding : _size_any_t | str = 0,
    dilation: _size_any_t  = 1,
    groups  : int          = 1
) -> torch.Tensor:
    """Applies 2D convolution with same padding.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].
        weight: Convolution kernel tensor as ``torch.Tensor`` with
            shape [C_out, C_in/groups, kH, kW].
        bias: Optional bias tensor as ``torch.Tensor`` with shape [C_out] or ``None``.
            Default is ``None``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding mode or size as ``int``, ``tuple[int, int]``, or ``str``.
            Default is ``0`` (updated by ``'pad_same'``).
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        groups: Number of groups in convolution as ``int``. Default is ``1``.

    Returns:
        Output tensor as ``torch.Tensor`` after convolution with same padding.
    """
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = weight.shape[-2:],
        stride      = stride,
        dilation    = dilation
    )
    y = F.conv2d(
        input    = y,
        weight   = weight,
        bias     = bias,
        stride   = stride,
        padding  = padding,
        dilation = dilation,
        groups   = groups
    )
    return y


class Conv2dSame(torch.nn.Conv2d):
    """2D convolution with TensorFlow-like SAME padding.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        kernel_size: Size of the convolution kernel as ``int`` or ``tuple[int, int]``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding size or mode as ``int``, ``tuple[int, int]``, or ``str``
            (overridden by SAME). Default is ``0``.
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        groups: Number of groups in convolution as ``int``. Default is ``1``.
        bias: Adds bias to convolution if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolution as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.

    Attributes:
        Inherits attributes from ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        groups      : int  = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies 2D convolution with SAME padding.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out]
            using SAME padding.
        """
        return conv2d_same(
            input    = input,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )


# ----- Conv + Norm + Act -----
ConvNormAct   = ConvNormActivation
Conv2dNormAct = Conv2dNormActivation
Conv3dNormAct = Conv3dNormActivation
