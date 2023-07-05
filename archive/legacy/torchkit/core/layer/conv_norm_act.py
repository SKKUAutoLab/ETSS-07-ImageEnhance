#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution + Normalization + Activation Layer.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
from torch import nn

from torchkit.core.utils import FuncCls
from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from torchkit.core.utils import to_2tuple
from .act import Mish
from .builder import CONV_NORM_ACT_LAYERS
from .conv import create_conv2d
from .norm_act import convert_norm_act
from .padding import autopad

logger = logging.getLogger()


# MARK: - ConvBnAct2d

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_act_2d")
class ConvBnAct2d(nn.Module):
    """Conv2d + BN + Act."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = "",
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = False,
        apply_act   : bool                     = True,
        norm_layer  : Optional[FuncCls]        = nn.BatchNorm2d,
        act_layer   : Optional[FuncCls]        = nn.ReLU,
        aa_layer    : Optional[FuncCls]        = None,
        drop_block  : Optional[FuncCls]        = None,
        *args, **kwargs
    ):
        super().__init__()
        use_aa      = aa_layer is not None
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        self.conv = create_conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = (1, 1) if use_aa else stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            **kwargs
        )

        # NOTE for backwards compatibility with models that use separate norm
        # and act layer definitions
        norm_act_layer = convert_norm_act(norm_layer, act_layer)
        self.bn = norm_act_layer(
            out_channels, apply_act=apply_act, drop_block=drop_block
        )
        self.aa = (aa_layer(channels=out_channels)
                   if stride == 2 and use_aa else None)

    # MARK: Properties

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


ConvBnAct = ConvBnAct2d
CONV_NORM_ACT_LAYERS.register(name="conv_bn_act", module=ConvBnAct)


# MARK: - ConvBnMish

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_mish_2d")
class ConvBnMish2d(nn.Sequential):
    """Conv2d + BN + Mish."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = None,
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = True,
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                **kwargs
            )
        )
        self.add_module("bn",  nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.Mish() if apply_act else nn.Identity())


ConvBnMish = ConvBnMish2d
CONV_NORM_ACT_LAYERS.register(name="conv_bn_mish", module=ConvBnMish)


# MARK: - ConvBnReLU

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu_2d")
class ConvBnReLU2d(nn.Sequential):
    """Conv2d + BN + ReLU."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = 0,
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = False,
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                **kwargs
            )
        )
        self.add_module("bn",  nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.ReLU() if apply_act else nn.Identity())


ConvBnReLU = ConvBnReLU2d
CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu", module=ConvBnReLU)


# MARK: - ConvBnReLU6

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu6_2d")
class ConvBnReLU62d(nn.Sequential):
    """Conv2d + BN + ReLU6."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (3, 3),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = None,
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = False,
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                **kwargs
            )
        )
        self.add_module("bn",  nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.ReLU6() if apply_act else nn.Identity())


ConvBnReLU6 = ConvBnReLU62d
CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu6", module=ConvBnReLU6)


# MARK: - CrossConv

@CONV_NORM_ACT_LAYERS.register(name="cross_conv")
class CrossConv(nn.Module):
    """Cross Convolution Downsample."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int   = 3,
        stride      : int   = 1,
        group       : int   = 1,
        expansion   : float = 1.0,
        shortcut    : bool  = False
    ):
        super().__init__()
        c_ = int(out_channels * expansion)  # Hidden channels
        from torchkit.core.layer import ConvBnMish
        self.cv1 = ConvBnMish(in_channels, c_,  (1, kernel_size), (1, stride))
        self.cv2 = ConvBnMish(c_, out_channels, (kernel_size, 1), (stride, 1),
                              group=group)
        self.add = shortcut and in_channels == out_channels
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# MARK: - CrossConvCSP

@CONV_NORM_ACT_LAYERS.register(name="cross_conv_csp")
class CrossConvCSP(nn.Module):
    """Cross Convolution CSP."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        kernel_size : int   = 3,
        stride      : int   = 1,
        group       : int   = 1,
        expansion   : float = 0.5,
        shortcut    : bool  = False
    ):
        super().__init__()
        c_ = int(out_channels * expansion)  # Hidden channels
        from torchkit.core.layer import ConvBnMish
        self.cv1 = ConvBnMish(in_channels, c_, 1, 1)
        self.cv2 = nn.Conv2d(in_channels,  c_, (1, 1), (1, 1), bias=False)
        self.cv3 = nn.Conv2d(c_, c_, (1, 1), (1, 1), bias=False)
        self.cv4 = ConvBnMish(2 * c_, out_channels, 1, 1)
        self.bn  = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m   = nn.Sequential(*[CrossConv(c_, c_, 3, 1, group, 1.0, shortcut)
                                   for _ in range(number)])
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


C3 = CrossConvCSP
CONV_NORM_ACT_LAYERS.register(name="c3", module=C3)


# MARK: - GhostConv

@CONV_NORM_ACT_LAYERS.register(name="ghost_conv")
class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T = 1,
        stride      : Size2T = 1,
        group       : int    = 1,
        apply_act   : bool   = True
    ):
        super().__init__()
        from torchkit.core.layer import ConvBnMish
        c_ = out_channels // 2  # Hidden channels
        self.cv1 = ConvBnMish(in_channels, c_, kernel_size, stride, group,
                              apply_act=apply_act)
        self.cv2 = ConvBnMish(c_, c_, 5, 1, c_, apply_act=apply_act)

    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)
