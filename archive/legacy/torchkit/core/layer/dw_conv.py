#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise Convolution Layers.
"""

from __future__ import annotations

import logging
import math

from torchkit.core.utils import Size2T
from .conv_norm_act import ConvBnMish

logger = logging.getLogger()


# MARK: - DWConv

def DWConv(
    in_channels : int,
    out_channels: int,
    kernel_size : Size2T = (1, 1),
    stride      : Size2T = (1, 1),
    apply_act   : bool   = True
):
    # Depthwise convolution
    return ConvBnMish(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = kernel_size,
        stride       = stride,
        groups       = math.gcd(in_channels, out_channels),
        apply_act    = apply_act
    )
