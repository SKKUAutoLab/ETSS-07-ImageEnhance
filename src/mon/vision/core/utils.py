#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions such as type checking, type casting,
dimension manipulation, etc.
"""

from __future__ import annotations

__all__ = [

]

import copy
from typing import Any

import numpy as np
import torch


def is_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is in the channel-first format. It is assumed
    that if the first dimension is the smallest.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
        )
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    return False


def is_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Return `True` if an image is in the channel-first format."""
    return not is_channel_first(image=image)


def get_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An image in channel-last or channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(
            f"image's number of dimensions must be between 2 and 4, but got "
            f"{image.ndim}."
        )
    if image.ndim == 4:
        if is_channel_first(image=image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
    elif image.ndim == 3:
        if is_channel_first(image=image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
    else:
        c = 1
    return c


def to_3d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D or 4-D image to a 3-D.

    Args:
        image: An image in channel-first format.

    Return:
        A 3-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 4, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 1HW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 1HW
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_list_of_3d(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a list of 3-D images.
   
    Args:
        image: An image of arbitrary type.
        
    Return:
        A list of 3-D images.
    """
    if isinstance(image, (torch.Tensor, np.ndarray)):
        if image.ndim == 3:
            image = [image]
        elif image.ndim == 4:
            image = list(image)
        else:
            raise ValueError
    elif isinstance(image, list | tuple):
        if not all(isinstance(i, (torch.Tensor, np.ndarray)) for i in image):
            raise ValueError
    return image


def to_4d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, or 5-D image to a 4-D.

    Args:
        image: An image in channel-first format.

    Return:
        A 4-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 5:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 5, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 11HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1NCHW -> NCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 11HW
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1NCHW -> NHWC
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_5d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, 4-D, or 6-D image to a 5-D.
    
    Args:
        image: An tensor in channel-first format.

    Return:
        A 5-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 6:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 6, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 111HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 11CHW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4:  # NCHW -> 1NCHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 6 and image.shape[0] == 1:  # 1*NCHW -> *NCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 111HW
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # HWC -> 11HWC
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 4:  # BHWC -> 1BHWC
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 6 and image.shape[0] == 1:  # 1*BHWC -> *BHWC
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(
            f"x must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_channel_first(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format.
    
    Args:
        image: An image in channel-last or channel-first format.
    
    Returns:
        An image in channel-first format.
    """
    if is_channel_first(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 5:
            image = image.permute(0, 1, 4, 2, 3)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 4, 2, 3))
    else:
        raise TypeError(
            f"image must be torch.Tensor or a numpy.ndarray, but got {type(image)}."
        )
    return image


def to_channel_last(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format.

    Args:
        image: An image in channel-last or channel-first format.

    Returns:
        A image in channel-last format.
    """
    if is_channel_last(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)
        elif image.ndim == 5:
            image = image.permute(0, 1, 3, 4, 2)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 3, 4, 2))
    else:
        raise TypeError(
            f"image must be torch.Tensor or a numpy.ndarray, but got {type(image)}."
        )
    return image
