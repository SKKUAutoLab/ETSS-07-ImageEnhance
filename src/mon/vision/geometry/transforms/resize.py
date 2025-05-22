#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements transformation functions."""

__all__ = [
    "pair_downsample",
    "resize",
]

from typing import Literal

import cv2
import kornia
import numpy as np
import torch

from mon.nn import _size_2_t, functional as F
from mon.vision import types


def pair_downsample(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample an image into a pair to half resolution.
    
    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W].

    Returns:
        Two downsampled images as ``tuple[torch.Tensor, torch.Tensor]``,
        each [B, C, H/2, W/2].

    Notes:
        Averages diagonal pixels in non-overlapping patches:
            ---------------------        ---------------------
            | A1 | B1 | A2 | B2 |        | A1+D1/2 | A2+D2/2 |
            | C1 | D1 | C2 | D2 |        | A3+D3/2 | A4+D4/2 |
            ---------------------  ===>  ---------------------
            | A3 | B3 | A4 | B4 |        | B1+C1/2 | B2+C2/2 |
            | C3 | D3 | C4 | D4 |        | B3+C3/2 | B4+C4/2 |
            ---------------------        ---------------------

    References:
        - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing
    """
    c       = image.shape[1]
    filter1 = torch.Tensor([[[[0, 0.5], [0.5, 0]]]]).to(image.dtype).to(image.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.Tensor([[[[0.5, 0], [0, 0.5]]]]).to(image.dtype).to(image.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


# noinspection PyTypeHints
def resize(
    image        : torch.Tensor | np.ndarray,
    size         : _size_2_t = None,
    divisible_by : int       = None,
    side         : Literal["short", "long", "vert", "horz", None] = None,
    interpolation: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area",
                           cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR] = "bilinear",
    **kwargs,
) -> torch.Tensor | np.ndarray:
    """Resize an image
    
    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W], range [0.0, 1.0], or
            ``np.ndarray`` in [H, W, C], range [0, 255].
        size: Target size as ``int`` or ``Sequence[int]``. Default is ``None``.
        divisible_by: If not ``None``, then the image will be resized to a size that is
            divisible by this number. Default: ``None``.
        side: Side to scale if ``size`` is ``int``. One of:
            - ``"short"``: Resize based on the shortest dimension.
            - ``"long"``: Resize based on the longest dimension.
            - ``"vert"``: Resize based on the vertical dimension.
            - ``"horz"``: Resize based on the horizontal dimension.
            Defaults is ``None``.
        interpolation: Upsampling method.
            - For ``kornia``:
                - ``"nearest"``
                - ``"linear"``
                - ``"bilinear"``
                - ``"bicubic"``
                - ``"trilinear"``
                -  ``"area"``
                Defaults is ``"bilinear"``.
            - For ``cv2``:
                - ``cv2.INTER_AREA``: This is used when we need to shrink an image.
                - ``cv2.INTER_CUBIC``: This is slow but more efficient.
                - ``cv2.INTER_LINEAR``: This is primarily used when zooming is required.
                    This is the default interpolation technique in OpenCV.
        **kwargs (korina.geometry.transform.resize):
            - align_corners: interpolation flag.
            - antialias: if ``True``, then image will be filtered with Gaussian
                before downscaling. No effect for upscaling.
        
        **kwargs (cv2.resize):
            - fx: Scale factor along the horizontal axis.
            - fy: Scale factor along the vertical axis.
            - antialias: If ``True``, then image will be filtered with Gaussian before
                downscaling. No effect for upscaling.
    
    Returns:
        Resized image as ``torch.Tensor`` or ``np.ndarray`` matching input type.

    Raises:
        TypeError: If image is not a ``torch.Tensor`` or ``np.ndarray``.
    """
    # Parse size
    if size:
        size = types.image_size(size, divisible_by)
    else:
        size = types.image_size(image, divisible_by)
        
    # Resize based on the shortest dimension
    if side == "short":
        h0, w0 = types.image_size(image)
        h1, w1 = size
        if h0 < w0:
            scale = h1 / h0
            new_h = h1
            new_w = int(w0 * scale)
        elif h0 > w0:
            scale = w1 / w0
            new_h = int(h0 * scale)
            new_w = w1
        else:
            scale = h1 / h0 if h1 < w1 else w1 / w0
            new_h = int(h0 * scale)
            new_w = int(w0 * scale)
        size = (new_h, new_w)
    # Resize based on the longest dimension
    elif side == "long":
        h0, w0 = types.image_size(image)
        h1, w1 = size
        if h0 > w0:
            scale = h1 / h0
            new_h = h1
            new_w = int(w0 * scale)
        elif h0 < w0:
            scale = w1 / w0
            new_h = int(h0 * scale)
            new_w = w1
        else:
            scale = h1 / h0 if h1 > w1 else w1 / w0
            new_h = int(h0 * scale)
            new_w = int(w0 * scale)
        size = (new_h, new_w)
    
    # Parse interpolation
    if isinstance(image, torch.Tensor):
        if interpolation in [cv2.INTER_AREA]:
            interpolation = "area"
        elif interpolation in [cv2.INTER_CUBIC]:
            interpolation = "bicubic"
        elif interpolation in [cv2.INTER_LINEAR]:
            interpolation = "linear"
    elif isinstance(image, np.ndarray):
        if interpolation in ["area"]:
            interpolation = cv2.INTER_AREA
        elif interpolation in ["bicubic"]:
            interpolation = cv2.INTER_CUBIC
        elif interpolation in ["linear", "bilinear", "trilinear", "nearest"]:
            interpolation = cv2.INTER_LINEAR
    
    # Apply the transformation
    if isinstance(image, torch.Tensor):
        align_corners = kwargs.pop("align_corners", None)
        antialias     = kwargs.pop("antialias",     False)
        return kornia.geometry.transform.resize(
            input         = image,
            size          = size,
            interpolation = interpolation,
            align_corners = align_corners,
            side          = side or "short",
            antialias     = antialias,
        )
    elif isinstance(image, np.ndarray):
        fx = kwargs.pop("fx", None)
        fy = kwargs.pop("fy", None)
        return cv2.resize(
            src   = image,
            dsize = (size[1], size[0]),
            fx    = fx,
            fy    = fy,
            interpolation = interpolation,
        )
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
