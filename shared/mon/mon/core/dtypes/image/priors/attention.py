#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements attention priors (saliency and focus maps).

This category includes methods that highlight important regions in an image, often used
in deep learning to guide processing or interpretation (e.g., attention maps in neural
networks).
"""

__all__ = [
    "BrightnessAttentionMap",
    "brightness_attention_map",
]

import kornia
import torch
import torch.nn as nn


def brightness_attention_map(
    image      : torch.Tensor,
    gamma      : float = 2.5,
    kernel_size: int   = None
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the enhancement
    network. Brighter regions are given lower weights to avoid over-saturation,
    while preserving image details and enhancing contrast in dark regions effectively.

    Args:
        image: An RGB image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
        gamma: Parameter controlling the curvature of the map. Default: ``2.5``.
        kernel_size: Window size for denoising operation. Default: ``None``.

    Returns:
        Brightness enhancement map with similar type and format as the input ``image``.
    """
    if kernel_size:
        image = kornia.filters.median_blur(image, kernel_size)
        # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        
    hsv = kornia.color.rgb_to_hsv(image)
    v   = hsv[:, 2:3, :, :]  # Extract the V-channel (brightness)
    bam = torch.pow((1 - v), gamma)
    return bam


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the enhancement
    network. Brighter regions are given lower weights to avoid over-saturation,
    while preserving image details and enhancing contrast in dark regions effectively.

    Args:
        gamma: Parameter controlling the curvature of the map. Default: ``2.5``.
        kernel_size: Window size for denoising operation. Default: ``None``.
    """
    
    def __init__(self, gamma: float = 2.5, kernel_size: int = None):
        super().__init__()
        self.gamma       = gamma
        self.kernel_size = kernel_size
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return brightness_attention_map(image, self.gamma, self.kernel_size)
