#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution with Weight Standardization (StdConv and ScaledStdConv).

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan
               Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in
unnormalized ResNets` - https://arxiv.org/abs/2101.08692
Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from torchkit.core.utils import to_2tuple
from .builder import CONV_LAYERS
from .padding import get_padding
from .padding import get_padding_value
from .padding import pad_same

logger = logging.getLogger()


# MARK: - StdConv2d

@CONV_LAYERS.register(name="std_conv_2d")
class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization` - https://arxiv.org/abs/1903.10520v2
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int                      = 1,
        padding     : Union[Size4T, str, None] = None,
        dilation    : int                      = 1,
        groups      : int                      = 1,
        bias        : bool                     = False,
        eps         : float                    = 1e-6
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias
        )
        self.eps = eps

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps
        ).reshape_as(self.weight)
        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )
        return x


StdConv = StdConv2d
CONV_LAYERS.register(name="std_conv", module=StdConv)


# MARK: - StdConv2dSame

@CONV_LAYERS.register(name="std_conv_2d_same")
class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for
     ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization` - https://arxiv.org/abs/1903.10520v2
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = "same",
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = False,
        eps         : float                    = 1e-6
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias
        )
        self.same_pad = is_dynamic
        self.eps      = eps

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps
        ).reshape_as(self.weight)
        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )
        return x


StdConvSame = StdConv2dSame
CONV_LAYERS.register(name="std_conv_same", module=StdConvSame)


# MARK: - ScaledStdConv2d

@CONV_LAYERS.register(name="scaled_std_conv_2d")
@CONV_LAYERS.register(name="ScaledStdConv2d")
class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in
    unnormalized ResNets` - https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind
    Haiku impl. The impact is minor.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int                      = 1,
        padding     : Union[Size4T, str, None] = None,
        dilation    : int                      = 1,
        groups      : int                      = 1,
        bias        : bool                     = True,
        gamma       : float                    = 1.0,
        eps         : float                    = 1e-6,
        gain_init   : float                    = 1.0
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias
        )
        self.gain  = nn.Parameter(
            torch.full((self.out_channels, 1, 1, 1), gain_init)
        )
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps   = eps

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            weight   = (self.gain * self.scale).view(-1),
            training = True,
            momentum = 0.0,
            eps      = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


ScaledStdConv = ScaledStdConv2d
CONV_LAYERS.register(name="scaled_std_conv", module=ScaledStdConv)


# MARK: - ScaledStdConv2dSame

@CONV_LAYERS.register(name="scaled_std_conv_2d_same")
class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME
    padding support

    Paper: `Characterizing signal propagation to close the performance gap in
    unnormalized ResNets` - https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind
    Haiku impl. The impact is minor.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = "same",
        dilation    : Size2T                   = (1, 1),
        groups      : int                      = 1,
        bias        : bool                     = True,
        gamma       : float                    = 1.0,
        eps         : float                    = 1e-6,
        gain_init   : float                    = 1.0
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias
        )
        self.gain = nn.Parameter(
            torch.full((self.out_channels, 1, 1, 1), gain_init)
        )
        self.scale    = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps      = eps

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            weight   = (self.gain * self.scale).view(-1),
            training = True,
            momentum = 0.0,
            eps      = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


ScaledStdConvSame = ScaledStdConv2dSame
CONV_LAYERS.register(name="scaled_std_conv_same", module=ScaledStdConvSame)
