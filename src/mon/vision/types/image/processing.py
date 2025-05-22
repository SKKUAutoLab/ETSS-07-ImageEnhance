#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
    - Image transformations.
    - Pixel operations.
"""

__all__ = [
    "add_images_weighted",
    "blend_images",
    "denormalize_image",
    "image_to_2d",
    "image_to_3d",
    "image_to_4d",
    "image_to_array",
    "image_to_channel_first",
    "image_to_channel_last",
    "image_to_tensor",
    "normalize_image",
    "normalize_image_by_range",
    "split_image",
]

import functools
from typing import Any
import math
import numpy as np
import torch

from mon.vision.types.image import utils


# ----- Fuse -----
def add_images_weighted(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    beta  : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Calculates the weighted sum of two image tensors.

    Args:
        image1: First image as ``torch.Tensor`` or ``numpy.ndarray``.
        image2: Second image as ``torch.Tensor`` or ``numpy.ndarray``.
        alpha: Weight for ``image1``.
        beta: Weight for ``image2``.
        gamma: Scalar offset added to the sum. Default is ``0.0``.
    
    Returns:
        Weighted sum as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Raises:
        ValueError: If ``image1`` and ``image2`` differ in shape or type.
        TypeError: If output type is not ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if image1.shape != image2.shape or type(image1) is not type(image2):
        raise ValueError(f"[image1] and [image2] must have the same shape and type, "
                         f"got {type(image1).__name__} and {type(image2).__name__}.")
    
    output = image1 * alpha + image2 * beta + gamma
    bound  = 1.0 if utils.is_image_normalized(image1) else 255.0
    
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound).astype(image1.dtype)
    else:
        raise TypeError(f"[output] must be a torch.Tensor or numpy.ndarray, got {type(output)}.")
    return output


def blend_images(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blends two images using a weighted sum.

    Args:
        image1: First image as ``torch.Tensor`` or ``numpy.ndarray``.
        image2: Second image as ``torch.Tensor`` or ``numpy.ndarray``.
        alpha: Weight for ``image1``, with ``image2`` weighted as (1 - ``alpha``).
        gamma: Scalar offset added to the sum. Default is ``0.0``.
    
    Returns:
        Blended image as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    return add_images_weighted(image1=image1, image2=image2, alpha=alpha, beta=1.0 - alpha, gamma=gamma)


# ----- Split -----
def split_image(image: torch.Tensor | np.ndarray, n: int = 2) -> list[np.ndarray]:
    """Split an image into ``n`` equal parts.

    Args:
        image: Image as ``numpy.ndarray`` [H, W, C].
        n: Number of parts to split into (positive integer). Default is ``2``.

    Returns:
        List of sub-images.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"[image] must be a 3D numpy array [H, W, C], got {image.shape}.")
    if n < 1:
        raise ValueError(f"[n] must be a positive integer, got {n}.")

    h, w = utils.image_size(image)
    if n > h * w:
        raise ValueError(f"[n] ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))

    return sub_images


# ----- Normalize -----
def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0
) -> torch.Tensor | np.ndarray:
    """Normalizes an image from range [min, max] to [new_min, new_max].

    Args:
        image: Image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        min: Current minimum pixel value. Default is ``0.0``.
        max: Current maximum pixel value. Default is ``255.0``.
        new_min: New minimum pixel value. Default is ``0.0``.
        new_max: New maximum pixel value. Default is ``1.0``.
    
    Returns:
        Normalized image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Raises:
        ValueError: If ``image`` dimensions are less than 3.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if not image.ndim >= 3:
        raise ValueError(f"[image]'s number of dimensions must be >= 3, got {image.ndim}.")
    
    ratio = (new_max - new_min) / (max - min)
    if isinstance(image, torch.Tensor):
        image = image.clone().to(dtype=torch.get_default_dtype())
    elif isinstance(image, np.ndarray):
        image = np.copy(image).astype(np.float32)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    image = (image - min) * ratio + new_min
    
    return image


normalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 255.0,
    new_min = 0.0,
    new_max = 1.0
)
denormalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 1.0,
    new_min = 0.0,
    new_max = 255.0
)


# ----- Convert -----
def image_to_2d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a 3D or 4D image to 2D.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        2D image as ``torch.Tensor`` [H, W] or ``numpy.ndarray`` [H, W].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    '''
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, "
                         f"got {image.ndim}.")
    '''
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.squeeze(0)
        elif image.ndim == 4 and image.shape[:2] == (1, 1):
            image = image.squeeze(0).squeeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = np.squeeze(image, -1)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 1:
            image = np.squeeze(image, (0, -1))
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def image_to_3d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a 2D or 4D image to 3D.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 2D or 4D format.
    
    Returns:
        3D image as ``torch.Tensor`` [C, H, W] or ``numpy.ndarray`` [H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 2, 3, or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    '''
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 2 and 4, "
                         f"got {image.ndim}.")
    '''
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            image = image.squeeze(1) if image.shape[1] == 1 else image.squeeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 4 and image.shape[0] == 1:
            image = np.squeeze(image, 0)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def image_to_4d(
    image: torch.Tensor | np.ndarray
           | list[torch.Tensor]       | list[np.ndarray]
           | tuple[torch.Tensor, ...] | tuple[np.ndarray, ...]
) -> torch.Tensor | np.ndarray:
    """Converts a 2D or 3D image to 4D.

    Args:
        image: Image as ``torch.Tensor``, ``numpy.ndarray``, or list/tuple of 3D/4D images.
    
    Returns:
        4D image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [B, H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 2, 3, or 4.
        TypeError: If ``image`` type is not supported.
    """
    '''
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 2 and 4, "
                         f"got {image.ndim}.")
    '''
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # [H, W] -> [1, 1, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:  # [C, H, W] -> [1, C, H, W]
            image = image.unsqueeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # [H, W] -> [1, H, W, 1]
            image = np.expand_dims(image, axis=(0, -1))
        elif image.ndim == 3:  # [H, W, C] -> [1, H, W, C]
            image = np.expand_dims(image, axis=0)
    elif isinstance(image, (list, tuple)):
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in image):
            image = torch.stack(image, dim=0)  # Stack 3D tensors to [B, C, H, W]
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
            image = torch.cat(image, dim=0)  # Concatenate 4D tensors along batch
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
            image = np.array(image)  # Convert list of 3D arrays to [B, H, W, C]
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
            image = np.concatenate(image, axis=0)  # Concatenate 4D arrays along batch
        else:
            raise TypeError(f"[image] list/tuple must contain consistent 3D or 4D "
                            f"torch.Tensor or numpy.ndarray, got mixed types or "
                            f"dimensions: "
                            f"{[type(i) for i in image]} "
                            f"{[i.shape for i in image if i is not None]}.")
    else:
        raise TypeError(f"[image] must be a torch.Tensor, numpy.ndarray, or "
                        f"list/tuple of either, got {type(image)}.")
    
    return image


def image_to_channel_first(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts an image to channel-first format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        Channel-first image as ``torch.Tensor`` [C, H, W] or [B, C, H, W], or
            ``numpy.ndarray`` [C, H, W] or [B, C, H, W].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if utils.is_image_channel_first(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, "
                         f"got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def image_to_channel_last(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts an image to channel-last format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        Channel-last image as ``torch.Tensor`` [H, W, C] or [B, H, W, C], or
            ``numpy.ndarray`` [H, W, C] or [B, H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if utils.is_image_channel_last(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def image_to_array(image: torch.Tensor | np.ndarray, denormalize: bool = False) -> np.ndarray:
    """Converts an image to a ``numpy.ndarray``.
    
    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        denormalize: Convert to [0, 255] range if ``True``. Default is ``True``.
    
    Returns:
        Image as ``numpy.ndarray`` in [H, W, C] or original shape if ``keepdim`` is ``True``.
    
    Raises:
        ValueError: If ``image`` dimensions are not 3, or 4.
        
    Recommend order:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    # Check shape
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    # Remove batch dimension
    image = image_to_3d(image)
    # Detach
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
    # Clamp
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, 1)
    else:
        image = np.clip(image, 0, 1)
    # Rearrange
    image = image_to_channel_last(image)
    # Convert to numpy
    image = image.numpy() if isinstance(image, torch.Tensor) else image
    # Denormalize
    if denormalize:
        image = denormalize_image(image).round().astype(np.uint8)
    
    return image


def image_to_tensor(
    image    : torch.Tensor | np.ndarray,
    normalize: bool = False,
    device   : Any  = None
) -> torch.Tensor:
    """Converts an image to a ``torch.Tensor`` with optional normalization.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
    
    Returns:
        Image as ``torch.Tensor`` in [B, C, H, W] format.
    
    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
        
    Recommend order:
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    """
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
        
    # Rearrange before sending to GPU for better memory layout.
    image = image_to_channel_first(image)
    # Ensure float32 for model input.
    image = image.float()
    # Normalize image
    image = normalize_image(image) if normalize else image
    # Add batch dimension
    image = image_to_4d(image)
    # Place on device
    if device:
        image = image.to(device)
    image = image.contiguous()
    
    return image
