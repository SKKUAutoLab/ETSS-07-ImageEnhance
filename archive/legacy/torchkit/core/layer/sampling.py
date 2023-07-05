#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downsampling and upsampling layers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchkit.core.utils import Size2T
from .builder import SAMPLING_LAYERS

logger = logging.getLogger()


# MARK: - Downsample/Downsampling

@SAMPLING_LAYERS.register(name="downsample")
class Downsample(nn.Sequential):
    """
    
    Args:
        in_channels (int):
            The number of input channels.
        scale_factor (int):
            The scale factor. Default: `0`.
        mode (str, optional):
            The upsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        super().__init__()
        self.add_module(
            "upsample", nn.Upsample(
                scale_factor=0.5, mode=mode, align_corners=align_corners
            )
        )
        self.add_module(
            "conv", nn.Conv2d(
                in_channels, in_channels + scale_factor, kernel_size=1,
                stride=1, padding=0, bias=False
            )
        )


Downsampling = Downsample
SAMPLING_LAYERS.register(name="downsampling", module=Downsampling)


# MARK: - Upsample/Upsampling

@SAMPLING_LAYERS.register(name="upsample")
class Upsample(nn.Sequential):
    """

    Args:
        in_channels (int):
            The number of input channels.
        scale_factor (int):
            The scale factor. Default: `0`.
        mode (str, optional):
            The upsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        
        super().__init__()
        self.add_module(
            "upsample", nn.Upsample(
                scale_factor=2.0, mode=mode, align_corners=align_corners
            )
        )
        self.add_module(
            "conv", nn.Conv2d(
                in_channels + scale_factor, in_channels, kernel_size=1,
                stride=1, padding=0, bias=False)
        )


Upsampling = Upsample
SAMPLING_LAYERS.register(name="upsampling", module=Upsampling)


# MARK: - SkipUpsample/SkipUpsampling

@SAMPLING_LAYERS.register(name="skip_upsample")
class SkipUpsample(nn.Module):
    """

    Args:
        in_channels (int):
            The number of input channels.
        scale_factor (int):
            The scale factor. Default: `0`.
        mode (str, optional):
            The upsampling algorithm. One of: [`nearest`, `linear`, `bilinear`,
            `bicubic`, `trilinear`]. Default: `nearest`.
        align_corners (bool, optional):
            If `True`, the corner pixels of the input and output tensors are
            aligned, and thus preserving the values at those pixels. This
            only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int            = 0,
        mode         : str            = "bilinear",
        align_corners: Optional[bool] = False
    ):
        
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode=mode,
                        align_corners=align_corners),
            nn.Conv2d(in_channels + scale_factor, in_channels, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.
		
		Args:
			x (torch.Tensor):
				The input tensor.
			skip (torch.Tensor):
				The skip connection tensor.
				
		Returns:
			y_hat (torch.Tensor):
				The output tensor.
		"""
        y_hat = self.up(x)
        y_hat = y_hat + skip
        return y_hat


SkipUpsampling = SkipUpsample
SAMPLING_LAYERS.register(name="skip_upsampling", module=SkipUpsampling)


# MARK: - PixelShufflePack

@SAMPLING_LAYERS.register(name="pixel_shuffle")
class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer. This module packs `F.pixel_shuffle()`
    and a nn.Conv2d module together to achieve a simple upsampling with pixel
    shuffle.
    
    Args:
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels.
        scale_factor (int):
            Upsample ratio.
        upsample_kernel (int, tuple):
            Kernel size of the conv layer to expand the channels.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        scale_factor   : int,
        upsample_kernel: Size2T,
    ):
        
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels * scale_factor * scale_factor,
            kernel_size  = upsample_kernel,
            padding      = (upsample_kernel - 1) // 2
        )
        self.init_weights()

    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.upsample_conv(x)
        y_hat = F.pixel_shuffle(y_hat, self.scale_factor)
        return y_hat
