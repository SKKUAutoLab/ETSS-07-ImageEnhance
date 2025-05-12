#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic implicit neural representation (INR) layers."""

__all__ = [
    "SigmoidLayer",
    "TanhLayer",
    "ReLULayer",
]

import numpy as np
import torch
from torch.nn import functional as F


# ----- Utils -----
def create_coords(down_size: int) -> torch.Tensor:
    """Creates a coordinates grid.

    Args:
        down_size: Size of the square coordinates grid as ``int``.

    Returns:
        Tensor of shape [down_size, down_size, 2] with normalized coords as ``torch.Tensor``.
    """
    h, w   = down_size, down_size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    return torch.from_numpy(coords).float()


def create_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Extracts patches into channels of a tensor.

    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W] as ``torch.Tensor``.
        kernel_size: Size of square patches as ``int``. Default is ``1``.

    Returns:
        Tensor with patches in channels as ``torch.Tensor``,
        shape [H', W', K^2] or [B, H', W', K^2].
    """
    from mon import vision
    num_channels = vision.image_num_channels(image)
    kernel       = torch.zeros(kernel_size**2, num_channels, kernel_size, kernel_size, device=image.device)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i * kernel_size + j, :, i, j] = 1
    
    im_padded = torch.nn.ReflectionPad2d(kernel_size // 2)(image)
    extracted = F.conv2d(im_padded, kernel, padding=0)
    return torch.movedim(extracted, 1 if image.dim() == 4 else 0, -1)


def interpolate_image(image: torch.Tensor, down_size: int) -> torch.Tensor:
    """Resizes image to a new square resolution.

    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W] as ``torch.Tensor``.
        down_size: Target size for height and width as ``int``.

    Returns:
        Resized tensor as ``torch.Tensor`` of shape
        [C, down_size, down_size] or [B, C, down_size, down_size].
    """
    return F.interpolate(image, size=(down_size, down_size), mode="bicubic")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Applies Fourier feature embedding to input tensor.

    Args:
        p: Input tensor to embed as ``torch.Tensor``.
        B: Projection matrix as ``torch.Tensor`` or ``None``. Default is ``None``.

    Returns:
        Embedded tensor as ``torch.Tensor`` with sine and cosine features.
    """
    if B is None:
        return p
    x_proj = (2 * np.pi * p) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ----- Basic Activation Layers -----
class SigmoidLayer(torch.nn.Module):
    """Applies linear transformation with sigmoid activation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sigmoid.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        return self.act(self.linear(x))
    

class TanhLayer(torch.nn.Module):
    """Applies linear transformation with tanh activation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and tanh.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``
        """
        return self.act(self.linear(x))
    
    
class ReLULayer(torch.nn.Module):
    """Applies linear transformation with ReLU activation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.

    References:
        - https://github.com/vishwa91/wire/blob/main/modules/relu.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and ReLU.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        return self.act(self.linear(x))
