#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image descriptive statistics priors.

This category includes methods that compute statistical properties (e.g., mean,
variance, standard deviation) over local regions of an image.
"""

__all__ = [
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)`
            in :math:`[0.0, 1.0]`.
        patch_size: Size of the sliding window. Default: ``5``.

    Returns:
        Local mean with similar type and format as the input ``image``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)`
            in :math:`[0.0, 1.0]`.
        patch_size: Size of the sliding window. Default: ``5``.

    Returns:
        Local variance with similar type and format as the input ``image``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9,
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)`
            in :math:`[0.0, 1.0]`.
        patch_size: Size of the sliding window. Default: ``5``.
        eps: Small value to avoid division by zero in sqrt. Default: ``1e-9``.

    Returns:
        Local standard deviation with similar type and format as the input ``image``.
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window. Default: ``5``.
        eps: Small value to avoid division by zero in sqrt. Default: ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        return image_local_stddev(image, self.patch_size, self.eps)
