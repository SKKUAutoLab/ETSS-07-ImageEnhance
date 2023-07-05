#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution + Activation Layer.
"""

from __future__ import annotations

import logging
from typing import Union

from torch import nn

from torchkit.core.utils import FuncCls
from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from torchkit.core.utils import to_2tuple
from .act import create_act_layer
from .builder import CONV_ACT_LAYERS
from .padding import autopad

logger = logging.getLogger()


# MARK: - ConvAct

@CONV_ACT_LAYERS.register(name="conv_act_2d")
class ConvAct2d(nn.Sequential):
    """Conv2d + Act.

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
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
        act_layer (nn.Module, str, optional):
            The activation layer or the name to build the activation layer.
        inplace (bool):
            Perform in-place activation. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = None,
        groups      : int                      = 1,
        apply_act   : bool                     = True,
        act_layer   : Union[FuncCls]           = nn.ReLU,
        inplace     : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        act_layer   = create_act_layer(apply_act, act_layer, inplace)
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                groups       = groups,
                bias         = False,
                *args, **kwargs
            )
        )
        self.add_module("act", act_layer)


ConvAct = ConvAct2d
CONV_ACT_LAYERS.register(name="conv_act", module=ConvAct)


# MARK: - ConvMish2d

@CONV_ACT_LAYERS.register(name="conv_mish_2d")
class ConvMish2d(nn.Sequential):
    """Conv2d + Mish.

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
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = None,
        groups      : int                      = 1,
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                groups       = groups,
                bias         = False,
                *args, **kwargs
            )
        )
        self.add_module("act", nn.Mish() if apply_act else nn.Identity())


ConvMish = ConvMish2d
CONV_ACT_LAYERS.register(name="conv_mish",  module=ConvMish)


# MARK: - ConvReLU2d

@CONV_ACT_LAYERS.register(name="conv_relu_2d")
class ConvReLU2d(nn.Sequential):
    """Conv2d + ReLU.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Size2T):
            Size of the convolving kernel.
        stride (Size2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Size4T, str, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        padding_mode (str):
            Default: `zeros`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T,
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = 0,
        padding_mode: str                      = "zeros",
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                padding_mode = padding_mode,
                *args, **kwargs
            )
        )
        self.add_module(
            "relu", nn.ReLU(inplace=True) if apply_act else nn.Identity()
        )


ConvReLU = ConvReLU2d
CONV_ACT_LAYERS.register(name="conv_relu", module=ConvReLU)


# MARK: - ConvSigmoid

@CONV_ACT_LAYERS.register(name="conv_sigmoid_2d")
class ConvSigmoid2d(nn.Sequential):
    """Conv2d + Sigmoid.

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
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Size2T                   = (1, 1),
        stride      : Size2T                   = (1, 1),
        padding     : Union[Size4T, str, None] = None,
        groups      : int                      = 1,
        apply_act   : bool                     = True,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                groups       = groups,
                bias         = False,
                *args, **kwargs
            )
        )
        self.add_module("sigmoid", nn.Sigmoid() if apply_act else nn.Identity())


ConvSigmoid = ConvSigmoid2d
CONV_ACT_LAYERS.register(name="conv_sigmoid", module=ConvSigmoid)
