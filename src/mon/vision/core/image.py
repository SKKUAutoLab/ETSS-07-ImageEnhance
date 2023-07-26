#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic functions for image data.
"""

from __future__ import annotations

__all__ = [
    "add_weighted",
    "blend",
]

import copy
import functools
from typing import Any

import numpy as np
import torch

from mon import nn
from mon.foundation import error_console, math
from mon.vision.core.utils import (
    get_num_channels, is_channel_first,
    to_3d, to_4d, to_channel_first, to_channel_last,
)


# region Access

def is_color_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is a color image. It is assumed that the image
    has 3 or 4 channels.
    """
    if get_num_channels(image=image) in [3, 4]:
        return True
    return False


def is_integer_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is integer-encoded."""
    c = get_num_channels(image=image)
    if c == 1:
        return True
    return False


def is_normalized_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is normalized."""
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def is_one_hot_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is one-hot encoded."""
    c = get_num_channels(image=image)
    if c > 1:
        return True
    return False


def check_image_size(size: list[int], stride: int = 32) -> int:
    """If the input :param:`size` isn't a multiple of the :param:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: An image's size.
        stride: The stride of a network. Default: 32.
    
    Returns:
        A new size of the image.
    """
    size     = get_hw(size=size)
    size     = size[0]
    new_size = math.make_divisible(size, divisor=int(stride))
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as :math:`(x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image=image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    :math:`(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image=image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def get_image_size(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height and width value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first(image=image):
        return [image.shape[-2], image.shape[-1]]
    else:
        return [image.shape[-3], image.shape[-2]]


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An image
    """
    if is_channel_first(image=image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]


def get_hw(size: int | list[int]) -> list[int]:
    """Casts a size object to the standard :math:`[H, W]` format.

    Args:
        size: A size of an image, windows, or kernels, etc.
    
    Returns:
        A size in :math:`[H, W]` format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            if size[0] >= size[3]:
                size = size[0:2]
            else:
                size = size[1:3]
        elif len(size) == 1:
            size = [size[0], size[0]]
    elif isinstance(size, int | float):
        size = [size, size]
    return size

# endregion


# region Edit

def add_weighted(
    image1: torch.Tensor | np.ndarray,
    alpha : float,
    image2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image.
        alpha: The weight of the :param:`image1` elements.
        image2: The second image.
        beta: The weight of the :param:`image2` elements.
        gamma: A scalar added to each sum. Default: 0.0.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"The shape of x and y must be the same, but got {image1.shape} and "
            f"{image2.shape}."
        )
    bound = 1.0 if image1.is_floating_point() else 255.0
    image = image1 * alpha + image2 * beta + gamma
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, bound).to(image1.dtype)
    elif isinstance(image, np.ndarray):
        image = np.clip(image, 0, bound)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def blend(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :param:`image1` * alpha + :param:`image2` * beta + gamma

    Args:
        image1: A source image.
        image2: A n overlay image that we want to blend on top of :param:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: 0.0.

    Returns:
        Blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )


def denormalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Denormalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image in channel-first format.
        mean: A sequence of means for each channel. Default:
            [0.485, 0.456, 0.406].
        std: A sequence of standard deviations for each channel. Default:
            [0.229, 0.224, 0.225].
        eps: A scalar value to avoid zero divisions. Default: 1e-6.
        
    Returns:
        A denormalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.devices
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.devices)
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1) if std_inv.ndim == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def normalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image in channel-first format.
        mean: A sequence of means for each channel. Default:
            [0.485, 0.456, 0.406].
        std: A sequence of standard deviations for each channel. Default:
            [0.229, 0.224, 0.225].
        eps: A scalar value to avoid zero divisions. Default: 1e-6.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.devices
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.devices)
        std += eps
        
        mean = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std  = std.view(-1, 1, 1) if std.ndim == 1 else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range [:param:`min`, :param:`max`] to
    the [:param:`new_min`, :param:`new_max`].
    
    Args:
        image: An image.
        min: The current minimum pixel value of the image. Default: 0.0.
        max: The current maximum pixel value of the image. Default: 255.0.
        new_min: A new minimum pixel value of the image. Default: 0.0.
        new_max: A new minimum pixel value of the image. Default: 1.0.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    # if is_normalized_image(image=image):
    #     return image
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        image = image.astype(np.float32)
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = np.cip(image, new_min, new_max)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


denormalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 1.0,
    new_min = 0.0,
    new_max = 255.0
)
normalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 255.0,
    new_min = 0.0,
    new_max = 1.0
)


def to_image_nparray(
    image      : torch.Tensor | np.ndarray,
    keepdim    : bool = True,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :class:`numpy.ndarray`.
    
    Args:
        image: An image.
        keepdim: If `True`, keep the original shape. If `False`, convert it to
            a 3-D shape. Default: `True`.
        denormalize: If `True`, convert image to :math:`[0, 255]`.
            Default: `True`.

    Returns:
        An :class:`numpy.ndarray` image.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.detach()
        image = image.cpu().numpy()
    image = denormalize_image(image=image).astype(np.uint) if denormalize else image
    image = to_channel_last(image=image)
    if not keepdim:
        image = to_3d(image=image)
    return image


def to_image_tensor(
    image    : torch.Tensor | np.ndarray,
    keepdim  : bool = True,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :class:`PIL.Image` or :class:`numpy.ndarray` to
    :class:`torch.Tensor`. Optionally, convert :param:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An image in channel-last or channel-first format.
        keepdim: If `True`, keep the original shape. If `False`, convert it to
            a 4-D shape. Default: `True`.
        normalize: If True, normalize the image to [0, 1]. Default: `False`.
        device: The device to run the model on. If None, the default ``'cpu'``
            device is used.
        
    Returns:
        A :class:`torch.Tensor` image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    image = to_channel_first(image=image)
    if not keepdim:
        image = to_4d(image=image)
    image = normalize_image(image=image) if normalize else image
    # Place in memory
    image = image.contiguous()
    if device is not None:
        device = nn.select_device(device=device) \
            if not isinstance(device, torch.device) else device
        image = image.to(device)
    return image

# endregion
