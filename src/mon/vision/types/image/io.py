#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements input/output operations for images.

Common Tasks:
    - Load images from disk.
    - Save images to disk.
    - Batch I/O.
    - Metadata handling.
"""

__all__ = [
    "load_image",
    "read_image_shape",
    "write_image",
]

from typing import Any

import cv2
import numpy as np
import rawpy
import torch
import torchvision
from PIL import Image

from mon import core
from mon.vision.types.image import processing, utils


# ----- Read -----
def load_image(
    path     : core.Path,
    flags    : int  = cv2.IMREAD_COLOR,
    to_tensor: bool = False,
    normalize: bool = False,
    device   : Any  = None
) -> torch.Tensor | np.ndarray:
    """Loads an image from a file path using OpenCV.

    Args:
        path: Image file path as ``core.Path`` or ``str``.
        flags: OpenCV flag for reading the image. Default is ``cv2.IMREAD_COLOR``.
        to_tensor: Convert to ``torch.Tensor`` if ``True``. Default is ``False``.
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
    
    Returns:
        RGB or grayscale image as ``torch.Tensor`` [B, C, H, W] or
        ``numpy.ndarray`` [H, W, C].
    """
    path = core.Path(path)
    if path.is_raw_image_file():  # Read raw image
        image = rawpy.imread(str(path))
        image = image.postprocess()
    else:  # Read other types of image
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # [H, W] -> [H, W, 1] for grayscale
            image = np.expand_dims(image, axis=-1)
        if utils.is_image_colored(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if to_tensor:
        image = processing.image_to_tensor(image, normalize=normalize, device=device)
    
    return image


def read_image_shape(path: core.Path) -> tuple[int, int, int]:
    """Reads an image shape from a file path using PIL or rawpy.

    Args:
        path: Image file path as ``core.Path`` or ``str``.

    Returns:
        Tuple of (height, width, channels) in [H, W, C] format.

    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = core.Path(path)
    if path.is_raw_image_file():
        image = rawpy.imread(str(path)).raw_image_visible
        h, w = image.shape
        c = 3
    else:
        with Image.open(str(path)) as image:
            w, h = image.size
            mode = image.mode
            c = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, None)
            if c is None:
                raise ValueError(f"Unsupported image mode {mode}.")
    
    return h, w, c


# ----- Write -----
def write_image(path: core.Path, image: torch.Tensor | np.ndarray):
    """Writes an image to a file path.

    Args:
        path: Output file path as ``core.Path`` or ``str``.
        image: Image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
    
    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    path = core.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
