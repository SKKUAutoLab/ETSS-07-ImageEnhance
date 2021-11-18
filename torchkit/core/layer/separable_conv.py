#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise Separable Conv Modules.

Basic DWS convs. Other variations of DWS exist with batch norm or activations
between the DW and PW convs such as the Depthwise modules in MobileNetV2 /
EfficientNet and Xception.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
from torch import nn as nn

from torchkit.core.utils import FuncCls
from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from .builder import CONV_LAYERS
from .builder import CONV_NORM_ACT_LAYERS
from .conv import create_conv2d
from .norm_act import convert_norm_act

logger = logging.getLogger()


# MARK: - SeparableConvBnAct

@CONV_NORM_ACT_LAYERS.register(name="separable_conv_bn_act")
class SeparableConvBnAct(nn.Module):
    """Separable Conv w/ trailing Norm and Activation."""

    # MARK: Magic Function

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int,
        kernel_size       : Size2T                   = (3, 3),
        stride            : Size2T                   = (1, 1),
        padding           : Union[Size4T, str, None] = "",
        dilation          : Size2T                   = (1, 1),
        bias              : bool                     = False,
        channel_multiplier: float                    = 1.0,
        pw_kernel_size    : int                      = 1,
        norm_layer        : FuncCls                  = nn.BatchNorm2d,
        act_layer         : FuncCls                  = nn.ReLU,
        apply_act         : bool                     = True,
        drop_block        : Optional[FuncCls]        = None
    ):
        super().__init__()
        self.conv_dw = create_conv2d(
            in_channels  = in_channels,
            out_channels = int(in_channels * channel_multiplier),
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            depthwise    = True
        )
        self.conv_pw = create_conv2d(
            in_channels  = int(in_channels * channel_multiplier),
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            padding      = padding,
            bias         = bias
        )
        norm_act_layer = convert_norm_act(norm_layer, act_layer)
        self.bn        = norm_act_layer(out_channels, apply_act=apply_act,
                                        drop_block=drop_block)

    # MARK: Properties

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# MARK: - SeparableConv2d

@CONV_LAYERS.register(name="separable_conv_2d")
class SeparableConv2d(nn.Module):
    """Separable Conv."""

    # MARK: Magic Function

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int,
        kernel_size       : Size2T                   = (3, 3),
        stride            : Size2T                   = (1, 1),
        padding           : Union[Size4T, str, None] = "",
        dilation          : Size2T                   = (1, 1),
        bias              : bool                     = False,
        channel_multiplier: float                    = 1.0,
        pw_kernel_size    : int                      = 1
    ):
        super().__init__()

        self.conv_dw = create_conv2d(
            in_channels  = in_channels,
            out_channels = int(in_channels * channel_multiplier),
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            depthwise    = True
        )
        self.conv_pw = create_conv2d(
            in_channels  = int(in_channels * channel_multiplier),
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            padding      = padding,
            bias         = bias
        )

    # MARK: Properties

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


SeparableConv = SeparableConv2d
CONV_LAYERS.register(name="separable_conv", module=SeparableConv)
