#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations that process image directly.

For example: basic structure, operations on image array, image filtering,
geometric image transformations, etc. We also define any augmentation
functions that contain an image here.
"""

from __future__ import annotations

import logging
import math
import random
import warnings
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
from multipledispatch import dispatch

from torchkit.core.data import ClassLabels
from torchkit.core.data import VisionData
from torchkit.core.utils import Dim3
from torchkit.core.utils import Image
from torchkit.core.utils import ImageDict
from torchkit.core.utils import ImageList
from torchkit.core.utils import ImageTuple

logger = logging.getLogger()


# MARK: - Basic Structures

@dispatch(torch.Tensor)
def is_channel_first(image: torch.Tensor) -> bool:
    """Check if the image is channel first layout."""
    # NOTE: 4D tensor
    if image.dim() == 4:
        _, s1, s2, s3 = [int(x) for x in image.shape]
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    # NOTE: 3D tensor
    elif image.dim() == 3:
        s0, s1, s2 = [int(x) for x in image.shape]
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    else:
        raise ValueError(f"Wrong image's dimension: {image.dim()}.")
    

@dispatch(np.ndarray)
def is_channel_first(image: np.ndarray) -> bool:
    """Check if the image is channel first layout."""
    # NOTE: 4D array
    if image.ndim == 4:
        _, s1, s2, s3 = image.shape
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    # NOTE: 3D array
    elif image.ndim == 3:
        s0, s1, s2 = image.shape
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    else:
        raise ValueError(f"Wrong image's dimension: {image.ndim}.")


def is_channel_last(image: Image) -> bool:
    """Check if the image is channel last layout."""
    return not is_channel_first(image)


@dispatch(torch.Tensor)
def to_channel_first(image: torch.Tensor) -> torch.Tensor:
    """Change image into channel first layout."""
    if is_channel_first(image):
        return image
    elif image.dim() == 3:
        return image.permute(2, 0, 1)
    elif image.dim() == 4:
        return image.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Wrong image's dimension, cannot reshape image: "
                         f"{image.dim()}.")


@dispatch(np.ndarray)
def to_channel_first(image: np.ndarray) -> np.ndarray:
    """Change image into channel first layout."""
    if is_channel_first(image):
        return image
    elif image.ndim == 3:
        return np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        return np.transpose(image, (0, 3, 1, 2))
    else:
        raise ValueError(f"Wrong image's dimension, cannot reshape image:"
                         f" {image.ndim}.")


@dispatch(torch.Tensor)
def to_channel_last(image: torch.Tensor) -> torch.Tensor:
    """Change image into channel last layout."""
    if is_channel_last(image):
        return image
    elif image.dim() == 3:
        return image.permute(1, 2, 0)
    elif image.dim() == 4:
        return image.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Wrong image's dimension, cannot reshape image: "
                         f"{image.dim()}.")


@dispatch(np.ndarray)
def to_channel_last(image: np.ndarray) -> np.ndarray:
    """Change image into channel last layout."""
    if is_channel_last(image):
        return image
    elif image.ndim == 3:
        return np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        return np.transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError(f"Wrong image's dimension, cannot reshape image: "
                         f"{image.ndim}.")


def to_pil_image(image: Image) -> PIL.Image:
    """Convert image from `np.ndarray` or `torch.Tensor` to PIL image."""
    if torch.is_tensor(image):
        # Equivalent to: ``np_image = image.numpy()`` but more efficient
        return torchvision.transforms.ToPILImage()(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype("uint8"), "RGB")
    
    raise TypeError(f"{type(image)} is not supported.")
    

@dispatch((np.ndarray, torch.Tensor), bool)
def reshape_image(
    image: Image, channel_first: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Change the image into channel first layout or channel last layout.
    """
    if channel_first:
        return to_channel_first(image)
    else:
        return to_channel_last(image)


@dispatch(list, bool)
def reshape_image(image: ImageList, channel_first: bool = False) -> ImageList:
    """Change the list of images into channel first layout or channel last
    layout.
    """
    assert isinstance(image, list)
    
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        image = reshape_image(np.array(image), channel_first)
        return list(image)
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        image = [reshape_image(i, channel_first) for i in image]
        return image
    
    # NOTE: List of torch.Tensor
    if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in image):
        image = reshape_image(torch.stack(image), channel_first)
        return list(image)
    if all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
        image = [reshape_image(i, channel_first) for i in image]
        return image
    
    raise TypeError(f"Cannot reshape images of type: {type(image)}.")


@dispatch(tuple, bool)
def reshape_image(image: ImageTuple, channel_first: bool = False) -> ImageTuple:
    """Change the tuple of images into channel first layout or channel last
    layout.
    """
    assert isinstance(image, tuple)
    image = list(image)
    image = reshape_image(image, channel_first)
    return tuple(image)
    

@dispatch(dict, bool)
def reshape_image(image: ImageDict, channel_first: bool = False) -> ImageDict:
    """Change the dict of images into channel first layout or channel last
    layout.
    """
    assert isinstance(image, dict)
    assert all(isinstance(v, (tuple, list, torch.Tensor, np.ndarray))
               for k, v in image.items())
    
    for k, v in image.items():
        image[k] = reshape_image(v, channel_first)
    
    return image


# MARK: - Operations on Arrays

@dispatch(torch.Tensor)
def is_integer_image(image: torch.Tensor):
    """Check if the given image is integer encoded."""
    max_value = torch.amax(image)
    min_value = torch.amin(image)
    if 255 >= max_value >= 0 and \
       255 >= min_value >= 0  and \
       (max_value - min_value) > 1:
        return True
    return False


@dispatch(np.ndarray)
def is_integer_image(image: np.ndarray):
    """Check if the given image is integer encoded."""
    max_value = np.amax(image)
    min_value = np.amin(image)
    if 255 >= max_value >= 0 and \
       255 >= min_value >= 0  and \
       (max_value - min_value) > 1:
        return True
    return False


@dispatch(torch.Tensor)
def is_normalized_image(image: torch.Tensor) -> bool:
    """Check if the image is given normalized."""
    max_value = torch.max(image)
    if 0 <= max_value <= 1.0:
        return True
    return False


@dispatch(np.ndarray)
def is_normalized_image(image: np.ndarray) -> bool:
    """Check if the given image is normalized."""
    max_value = np.max(image)
    if 0 <= max_value <= 1.0:
        return True
    return False


@dispatch(torch.Tensor)
def is_onehot_image(image: torch.Tensor):
    """Check if the given image is one-hot encoded."""
    max_value = torch.amax(image)
    min_value = torch.amin(image)
    if 1.0 >= max_value >= 0.0 and \
       1.0 >= min_value >= 0.0 and \
       (max_value - min_value) <= 1.0:
        return True
    return False


@dispatch(np.ndarray)
def is_onehot_image(image: np.ndarray):
    """Check if the given image is one-hot encoded."""
    max_value = np.amax(image)
    min_value = np.amin(image)
    if 1.0 >= max_value >= 0.0 and \
       1.0 >= min_value >= 0.0 and \
       (max_value - min_value) <= 1.0:
        return True
    return False


@dispatch(torch.Tensor, int)
def _to_onehot_image(image: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Change the given image to onehot format.

    Args:
        image (torch.Tensor):
            The integer encoded image.
        num_classes (int):
            The total number of classes.

    Returns:
        onehot (torch.Tensor):
            The onehot encoded image.
    """
    # NOTE: If the image is already in one-hot encoded
    if is_onehot_image(image):
        return image
    
    # NOTE: Convert to one-hot encoded mask
    # One-hot mask is of shape [CHW], where C == num_classes.
    # In each channel, the pixel = 1 if it is belong to the class's target and 0 for the rest.
    onehot = torch.nn.functional.one_hot(image, num_classes)
    onehot = onehot.transpose(1, 4).squeeze(-1)
    return onehot


@dispatch(torch.Tensor, int)
def to_onehot_image(image: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Change the given image to onehot format.

    Args:
        image (torch.Tensor):
            The integer encoded image. The image can be:
                - A 4D tensor of shape [BCHW].
                - A 3D image of shape [CHW].
        num_classes (int):
            The total number of classes.

    Returns:
        onehot (torch.Tensor):
            The onehot encoded image.
    """
    # NOTE: Convert to channel-first
    img = to_channel_first(image)
    
    # NOTE: If the mask is of shape [BCHW]
    if img.ndim == 4:
        ones = [_to_onehot_image(i, num_classes) for i in img]
        ones = torch.stack(ones).to(torch.uint8)
        return ones
    
    # NOTE: If the mask is of shape [CHW]
    if img.ndim == 3:
        return _to_onehot_image(img, num_classes)
    
    
@dispatch(np.ndarray, int)
def _to_onehot_image(image: np.ndarray, num_classes: int) -> np.ndarray:
    """Change the given image to one-hot format.

    Args:
        image (np.ndarray):
            The integer encoded image.
        num_classes (int):
            The total number of classes.

    Returns:
        onehot (np.ndarray):
            The onehot encoded image.
    """
    # NOTE: If the image is already in one-hot encoded
    if is_onehot_image(image):
        return image
    
    # NOTE: If the image is of shape [CHW], squeeze it to [HW]
    if image.ndim == 3:
        image = np.squeeze(image)
    assert image.ndim == 2, f"Image must be of shape [HW]."
    
    # NOTE: Convert to onehot encoded image
    # One-hot mask is of shape [CHW], where C == num_classes.
    # In each channel, the pixel = 1 if it is belong to the class's target and 0 for the rest.
    onehot = [np.isclose(image, i) for i in range(num_classes)]
    onehot = np.stack(onehot).astype(np.float32)
    return onehot


@dispatch(np.ndarray, int)
def to_onehot_image(image: np.ndarray, num_classes: int) -> np.ndarray:
    """Change the given image to one-hot encoded.

    Args:
        image (np.ndarray):
            The integer encoded image. The mask can be:
                - A 4D array of shape [BCHW].
                - A 3D image of shape [CHW].
        num_classes (int):
            The total number of classes.

    Returns:
        onehot (np.ndarray):
            The onehot encoded image(s).
    """
    # NOTE: Convert to channel-first
    img = to_channel_first(image)

    # NOTE: If image is of shape [BCHW]
    if img.ndim == 4:
        ones = [_to_onehot_image(i, num_classes) for i in img]
        ones = np.stack(ones).astype(np.uint8)
        return ones
    # NOTE: If the image is of shape [CHW]
    elif img.ndim == 3:
        return _to_onehot_image(img, num_classes)
    else:
        raise ValueError(f"Do not support image with ndim: {img.ndim}.")


def _to_integer_image(image: np.ndarray) -> np.ndarray:
    """Change the onehot encoded image to integer format."""
    # NOTE: If the mask is already in integer encoded
    if is_integer_image(image):
        return image
    
    # NOTE: Convert to integer encoded mask
    assert image.ndim == 3, "Image must be of shape [CHW]."
    integer = np.argmax(image, axis=0).astype(np.uint8)
    
    # NOTE: Expand dims to: [1HW]
    if integer.ndim == 2:
        integer = np.expand_dims(integer, axis=0)
    
    return integer


@dispatch(torch.Tensor)
def to_integer_image(image: torch.Tensor) -> torch.Tensor:
    """Change the one-hot encoded mask to integer format."""
    image_np = image.numpy()
    image_np = to_integer_image(image_np)
    return torch.from_numpy(image_np)


@dispatch(np.ndarray)
def to_integer_image(image: np.ndarray) -> np.ndarray:
    """Change the onehot encoded image to integer format."""
    # NOTE: Convert to channel-first
    img = to_channel_first(image)
    
    # NOTE: If the mask is of shape [CHW]
    if img.ndim == 3:
        return _to_integer_image(img)
    
    # NOTE: If the mask is of shape [BCHW]
    if img.ndim == 4:
        integers = [_to_integer_image(i) for i in image]
        integers = np.stack(integers, axis=0).astype(np.uint8)
        return integers
    
    raise ValueError("Wrong mask's dim, cannot encode mask.")


def _to_color_image(image: np.ndarray, colors: list) -> np.ndarray:
    """Fill an image with corresponding labels' colors.

    Args:
        image (np.ndarray):
            An image in either one-hot or integer.
        colors (list):
            The list of all colors.

    Returns:
        color (np.ndarray):
            The colored image.
    """
    # NOTE: Convert to integer encoded
    if is_onehot_image(image):
        image = to_integer_image(image)
    
    # NOTE: Squeeze dims to 2
    if image.ndim == 3:
        image = np.squeeze(image)
    
    # TODO: Draw color
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, len(colors)):
        idx    = image == l
        r[idx] = colors[l][0]
        g[idx] = colors[l][1]
        b[idx] = colors[l][2]
    rgb = np.stack([r, g, b], axis=0)
    return rgb


@dispatch(torch.Tensor, list)
def to_color_image(image: torch.Tensor, colors: list) -> torch.Tensor:
    """Fill mask with corresponding labels' colors.

    Args:
        image (torch.Tensor):
            An image in either one-hot or integer. The image can be:
                - A 4D array or tensor of shape [BCHW].
                - A 3D image of shape [CHW].
        colors (list):
            The list of all labels' colors.

    Returns:
        color (torch.Tensor):
            The colored image.
    """
    mask_np = image.numpy()
    mask_np = to_color_image(mask_np, colors)
    color   = torch.from_numpy(mask_np)
    return color


@dispatch(np.ndarray, list)
def to_color_image(image: np.ndarray, colors: list) -> np.ndarray:
    """Fill mask with corresponding labels' colors.

    Args:
        image (np.ndarray):
            An image in either one-hot or integer. The image can be:
                - A 3D image of shape [CHW].
                - A 4D array of shape [BCHW].
        colors (list):
            The list of all labels' colors.

    Returns:
        color (np.ndarray):
            The colored image.
    """
    assert len(colors) > 0, f"No colors are provided."
    
    # NOTE: Convert to channel-first
    img = to_channel_first(image)

    # NOTE: If the image is of shape [CHW]
    if img.ndim == 3:
        return _to_color_image(img, colors)
    
    # NOTE: If the img is of shape [BCHW]
    if img.ndim == 4:
        colors = [_to_color_image(i, colors) for i in img]
        colors = np.stack(colors).astype(np.uint8)
        return colors
    
    raise ValueError("Wrong mask's dim, cannot encode mask.")


@dispatch(torch.Tensor)
def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Convert `image` from float32/float16 in range [0.0, 1.0] to uint8 in
    range [0, 255].
    """
    # if is_normalized_image(image):
    image = torch.mul(image, 255)
    image = torch.clamp(image, min=0, max=255)
    return image


@dispatch(np.ndarray)
def unnormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert `image` from float32/float16 in range [0.0, 1.0] to uint8 in
    range [0, 255].
    """
    # if is_normalized_image(image):
    image = np.clip(image * 255.0, 0, 255.0).astype("uint8")
    return image


@dispatch(list)
def unnormalize_image(image: ImageList) -> ImageList:
    """Convert `image` from float32/float16 in range [0.0, 1.0] to uint8 in
    range [0, 255].
    """
    assert isinstance(image, list)
    
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        image = unnormalize_image(np.array(image))
        return list(image)
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        image = [unnormalize_image(i) for i in image]
        return image
    
    # NOTE: List of torch.Tensor
    if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in image):
        image = unnormalize_image(torch.stack(image))
        return list(image)
    if all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
        image = [unnormalize_image(i) for i in image]
        return image
    
    raise TypeError(f"Cannot unnormalize images of type: {type(image)}.")


@dispatch(tuple)
def unnormalize_image(image: ImageTuple) -> ImageTuple:
    """Convert `image` from float32/float16 in range [0.0, 1.0] to uint8 in
    range [0, 255].
    """
    assert isinstance(image, tuple)
    image = list(image)
    image = unnormalize_image(image)
    return tuple(image)
    

@dispatch(dict)
def unnormalize_image(image: ImageDict) -> ImageDict:
    """Convert `image` from float32/float16 in range [0.0, 1.0] to uint8 in
    range [0, 255].
    """
    assert isinstance(image, dict)
    assert all(isinstance(v, (tuple, list, torch.Tensor, np.ndarray))
               for k, v in image.items())
    
    for k, v in image.items():
        image[k] = unnormalize_image(v)
    
    return image


# MARK: - Image Filtering


# MARK: - Image Transformations

@dispatch(torch.Tensor, torch.Tensor, float, float)
def blend_images(
    overlays: torch.Tensor,
    images  : torch.Tensor,
    alpha   : float,
    gamma   : float = 0.0
) -> torch.Tensor:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (torch.Tensor):
            The images we want to overlay on top of the original image.
        images (torch.Tensor):
            The source images.
        alpha (float):
            The alpha transparency of the overlay.
        gamma (float):

    Returns:
        blend (torch.Tensor):
            The blended image.
    """
    overlays_np = overlays.numpy()
    images_np   = images.numpy()
    blends      = blend_images(overlays_np, images_np, alpha, gamma)
    blends      = torch.from_numpy(blends)
    return blends


@dispatch(np.ndarray, np.ndarray, float, float)
def blend_images(
    overlays: np.ndarray,
    images  : np.ndarray,
    alpha   : float,
    gamma   : float = 0.0
) -> Optional[np.ndarray]:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (np.ndarray):
            The images we want to overlay on top of the original image.
        images (np.ndarray):
            The source images.
        alpha (float):
            The alpha transparency of the overlay.
        gamma (float):

    Returns:
        blend (np.ndarray, optional):
            The blended image.
    """
    # NOTE: Type checking
    assert overlays.ndim == images.ndim, \
        f"image1 dims != image2 dims: {overlays.ndim} != {images.ndim}"
    
    # NOTE: Convert to channel-first
    overlays = to_channel_first(overlays)
    images   = to_channel_first(images)
    
    # NOTE: Unnormalize images
    images = unnormalize_image(images)
    
    # NOTE: Convert overlays to same data type as images
    images   = images.astype(np.uint8)
    overlays = overlays.astype(np.uint8)
    
    # NOTE: If the images are of shape [CHW]
    if overlays.ndim == 3 and images.ndim == 3:
        return cv2.addWeighted(overlays, alpha, images, 1.0 - alpha, gamma)
    
    # NOTE: If the images are of shape [BCHW]
    if overlays.ndim == 4 and images.ndim == 4:
        assert overlays.shape[0] == images.shape[0], \
            f"Number of batch in image1 != Number of batch in image2: " \
            f"{overlays.shape[0]} != {images.shape[0]}"
        blends = []
        for overlay, image in zip(overlays, images):
            blends.append(cv2.addWeighted(overlay, alpha, image, 1.0 - alpha,
                                          gamma))
        blends = np.stack(blends, axis=0).astype(np.uint8)
        return blends
    
    warnings.warn(f"Cannot blend images and overlays with dimensions: "
                  f"{images.ndim} and {overlays.ndim}")
    return None


@dispatch(torch.Tensor, int)
def concatenate_images(images: torch.Tensor, nrow: int = 1) -> torch.Tensor:
    """Concatenate multiple images into a single image.

    Args:
        images (torch.Tensor):
            The images can be:
                - A 4D mini-batch tensor of shape [BCHW].
                - A 3D RGB image of shape [CHW].
                - A 2D grayscale image of shape [HW].
        nrow (int):
            Number of images in each row of the grid. The final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (torch.Tensor):
            The concatenated image.
    """
    return torchvision.utils.make_grid(tensor=images, nrow=nrow)


@dispatch(np.ndarray, int)
def concatenate_images(images: np.ndarray, nrow: int = 1) -> np.ndarray:
    """Concatenate multiple images into a single image.

    Args:
        images (np.array):
            The images can be:
                - A 4D mini-batch tensor of shape [BCHW] or [BHWC].
                - A 3D RGB image of shape [CHW] or [HWC].
                - A 2D grayscale image of shape [HW].
        nrow (int):
            Number of images in each row of the grid. The final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (np.ndarray):
            The concatenated image.
    """
    # NOTE: Type checking
    if images.ndim == 3:
        return images
    
    # NOTE: Conversion (just for sure)
    if is_channel_last(images):
        images = to_channel_last(images)
    
    b, c, h, w = images.shape
    ncols      = nrow
    nrows      = (b // nrow) if (b // nrow) > 0 else 1
    cat_image  = np.zeros((c, int(h * nrows), w * ncols))
    for idx, im in enumerate(images):
        j = idx // ncols
        i = idx % ncols
        cat_image[:, j * h: j * h + h, i * w: i * w + w] = im
    return cat_image


@dispatch(list, int)
def concatenate_images(images: ImageList, nrow: int = 1) -> Image:
    """Concatenate multiple images into a single image.

    Args:
        images (list):
            A list of images all of the same shape [CHW].
        nrow (int):
            Number of images in each row of the grid. The final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (Image):
            The concatenated image.
    """
    if (isinstance(images, list) and
        all(isinstance(t, np.ndarray) for t in images)):
        cat_image = np.concatenate([images], axis=0)
        return concatenate_images(cat_image, nrow)
    elif isinstance(images, list) and all(torch.is_tensor(t) for t in images):
        return torchvision.utils.make_grid(tensor=images, nrow=nrow)
    else:
        raise TypeError(f"Cannot concatenate images of type: {type(images)}.")


@dispatch(dict, int)
def concatenate_images(images: ImageDict, nrow: int = 1) -> Image:
    """Concatenate multiple images into a single image.

    Args:
        images (dict):
            A dict of images all of the same shape [CHW].
        nrow (int, optional):
            Number of images in each row of the grid. The final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (Image):
            The concatenated image.
    """
    if (isinstance(images, dict) and
        all(isinstance(t, np.ndarray) for k, t in images.items())):
        cat_image = np.concatenate(
            [image for key, image in images.items()], axis=0
        )
        return concatenate_images(cat_image, nrow)
    elif (isinstance(images, dict) and
          all(torch.is_tensor(t) for k, t in images.items())):
        values = list(tuple(images.values()))
        return torchvision.utils.make_grid(values, nrow)
    else:
        raise TypeError(f"Cannot concatenate images of type: {type(images)}.")


def distort_image(
    image      : np.ndarray,
    orientation: str      = "horizontal",
    func       : Callable = np.sin,
    x_scale    : float    = 0.05,
    y_scale    : float    = 5.0
):
    """Distort image.

    Args:
        image (np.ndarray):
            The image.
        orientation (str):
            The orientation to distort the image.
        func (callable):
            Supported functions are `np.sin` and `np.cos`.
        x_scale (float):
            Scale in x direction.
        y_scale (float):
            Scale in y direction.

    Returns:
        image_distort (np.ndarray):
            The distorted image.
    """
    assert orientation[:3] in ["hor", "ver"], \
        "dist_orient should be 'horizontal'|'vertical'"
    assert func in [np.sin, np.cos], \
        "Supported functions are np.sin and np.cos"
    assert 0.00 <= x_scale <= 0.1, \
        "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(image.shape[0], image.shape[1]), \
        "y_scale should be less then image size"
    image_distort = image.copy()
    
    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))
    
    for c in range(3):
        for i in range(image.shape[orientation.startswith("ver")]):
            if orientation.startswith("ver"):
                image_distort[:, i, c] = np.roll(image[:, i, c], shift(i))
            else:
                image_distort[i, :, c] = np.roll(image[i, :, c], shift(i))
    
    return image_distort


def letterbox(
    image     : np.ndarray,
    new_shape : Union[int, Dim3] = 768,
    color     : tuple            = (114, 114, 114),
    auto      : bool             = True,
    scale_fill: bool             = False,
    scale_up  : bool             = True
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = image.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio     = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh    = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh    = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch
        dw, dh    = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio     = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image       = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return image, ratio, (dw, dh)


def random_perspective(
    image      : np.ndarray,
    rotate     : float    = 10,
    translate  : float    = 0.1,
    scale      : float    = 0.1,
    shear      : float    = 10,
    perspective: float    = 0.0,
    border     : Sequence = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask labels.

    Args:
        image (np.ndarray):
            The image.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image_new (np.ndarray):
            The augmented image.
        mask_labels_new (np.ndarray):
            The augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # a += random.choice([-180, -90, 0, 90])  # Add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    
    # NOTE: Translation
    T       = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # Image changed
        if perspective:
            image_new = cv2.warpPerspective(image_new, M, dsize=(width, height),
                                            borderValue=(114, 114, 114))
        else:  # Affine
            image_new = cv2.warpAffine(image_new, M[:2], dsize=(width, height),
                                       borderValue=(114, 114, 114))
    
    return image_new


def random_perspective_mask(
    image      : np.ndarray,
    mask       : np.ndarray = (),
    rotate     : float      = 10,
    translate  : float      = 0.1,
    scale      : float      = 0.1,
    shear      : float      = 10,
    perspective: float      = 0.0,
    border     : Sequence   = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask.

    Args:
        image (np.ndarray):
            The image.
        mask (np.ndarray):
            The mask.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image_new (np.ndarray):
            The augmented image.
        mask_new (np.ndarray):
            The augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    mask_new  = mask.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # a += random.choice([-180, -90, 0, 90])  # Add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    
    # NOTE: Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # Image changed
        if perspective:
            image_new = cv2.warpPerspective(image_new, M, dsize=(width, height),
                                            borderValue=(114, 114, 114))
            mask_new  = cv2.warpPerspective(mask_new, M, dsize=(width, height),
                                            borderValue=(114, 114, 114))
        else:  # Affine
            image_new = cv2.warpAffine(image_new, M[:2], dsize=(width, height),
                                       borderValue=(114, 114, 114))
            mask_new  = cv2.warpAffine(mask_new, M[:2], dsize=(width, height),
                                       borderValue=(114, 114, 114))
    
    return image_new, mask_new


def resize_image(
    image        : np.ndarray,
    new_shape    : Union[Dim3, int],
    interpolation: Optional[int] = None
) -> tuple[np.ndarray, tuple, tuple]:
    """Resize image using OpenCV functions.

    Args:
        image (np.ndarray):
            The image, should be of shape [HWC].
        new_shape (Dim3, int):
            The desired output size. If size is a sequence like [H, W, C],
            output size will be matched to this. If size is an int, larger
            edge of the image will be matched to this number. i.e,
            if height < width, then image will be rescaled to
            (size * width / height, size).
        interpolation (int, optional):
            The interpolation method.
        
    Returns:
        image (np.ndarray):
            The resized image.
        hw0 (tuple):
            The original HW.
        hw1 (tuple):
            The resized HW.
    """
    # NOTE: Convert to channel last to be sure
    new_image = image.copy()
    new_image = to_channel_last(new_image)
    
    # NOTE: Calculate hw0 and hw1
    h0, w0 = new_image.shape[:2]  # Original HW
    if isinstance(new_shape, int):
        ratio  = new_shape / max(h0, w0)  # Resize image to image_size
        h1, w1 = int(h0 * ratio), int(w0 * ratio)
    elif isinstance(new_shape, (tuple, list)):
        ratio  = 0 if (h0 != new_shape[0] or w0 != new_shape[1]) else 1
        h1, w1 = new_shape[:2]
    else:
        raise ValueError(f"Do not support new image shape of type: "
                         f"{type(new_shape)}")
    
    if interpolation is None:
        interpolation = cv2.INTER_AREA if (ratio < 1) else cv2.INTER_LINEAR
    
    # NOTE: Resize
    if ratio != 1:
        # Always resize down, only resize up if training with augmentation
        new_image = cv2.resize(
            src=new_image, dsize=(w1, h1), interpolation=interpolation
        )
 
    return new_image, (h0, w0), (h1, w1)


def scale_image(
    image: torch.Tensor, ratio: float = 1.0, same_shape: bool = False
) -> torch.Tensor:
    # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return image
    else:
        h, w = image.shape[2:]
        s    = (int(h * ratio), int(w * ratio))  # new size
        img  = F.interpolate(
            image, size=s, mode="bilinear", align_corners=False
        )  # Resize
        if not same_shape:  # Pad/crop img
            gs   = 128 #64 #32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        # Value = imagenet mean
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def padded_resize_image(
    image: np.ndarray, new_shape: Dim3
) -> np.ndarray:
    """Perform pad and resize image.

    Args:
        image (np.ndarray):
            The list of image or a single one.
        new_shape (Dim3):
            The desired size as [H, W, C].

    Returns:
        image (np.ndarray):
            The converted image.
    """
    if image.ndim == 4:
        image = [letterbox(img, new_shape=new_shape)[0] for img in image]
    else:
        image = letterbox(image, new_shape=new_shape)[0]

    # NOTE: Convert
    # image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # image = image[:, :, ::-1]  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)
    return image


# MARK: - Drawing Functions

def create_semantic_image(
    data       : VisionData,
    classlabels: ClassLabels,
    image      : Optional[np.ndarray] = None,
    encoding   : str                  = "color",
) -> np.ndarray:
    """Create semantic segmentation image from a list of instances.

    Args:
        data (VisionData):
            A `VisualData` object.
        classlabels (ClassLabels):
            The `ClassLabels` object contains all class-labels defined in the
            dataset.
        image (np.ndarray, optional):
            The image.
        encoding (str):
            The format to use when creating the semantic segmentation mask.
            Two most common ones are `color` and `id`. But can be other
            values depend on each dataset.
        
    Returns:
        semantic (np.ndarray):
            The semantic image.
    """
    # NOTE: Prepare data
    if isinstance(image, np.ndarray):
        h, w, c = image.shape
    else:
        h, w, c = data.image_info.shape0
        
    name2label = classlabels.name2label
    id2label   = classlabels.id2label
    
    # NOTE: The background color
    unlabeled = name2label.get("unlabeled")
    bg_color  = unlabeled.get(encoding)
    assert unlabeled is not None, \
        logger.error(f"classlabels doesn't have `unlabeled` label.")
    assert bg_color  is not None, \
        logger.error(f"Unknown encoding `{encoding}`.")
    
    # NOTE: This is the image that we want to create
    if encoding == "color":
        semantic = np.full((h, w, 3), bg_color, dtype=np.uint8)
    else:
        semantic = np.full((h, w, 1), bg_color, dtype=np.uint8)
    
    # NOTE: Loop over all objects
    for obj in data.objects:
        class_id = obj.class_id
        polygon  = obj.polygon
        deleted  = getattr(obj, "deleted", False)

        # NOTE: Get value
        # If the class ID is negative, the object is deleted, or the polygon
        # only has 3 vertices, skip it
        if class_id < 0 or deleted or len(polygon) < 3:
            continue
        # If the label is not known, flag error
        if class_id not in id2label:
            logger.info(f"Class ID `{class_id}` is unknown.")
            continue
        value = id2label[class_id].get(encoding)
        assert value is not None, \
            logger.error(f"Unknown encoding `{encoding}`.")
        
        # NOTE: Draw polygon
        cv2.fillPoly(semantic, pts=[polygon], color=value)

    return semantic


# MARK: - Color Space Conversions

def augment_hsv(
    image        : np.ndarray,
    hgain        : float = 0.5,
    sgain        : float = 0.5,
    vgain        : float = 0.5,
    equalize_hist: bool  = False
):
    """Augment HSV channels.

    Args:
        image (np.ndarray):
            The image.
        hgain (float):
            H-channel gains.
        sgain (float):
            S-channel gains.
        vgain (float):
            V-channel gains.
        equalize_hist (bool):
            Should equalize the histogram?

    Returns:
        image_augment (np.ndarray):
            The augmented image.
    """
    image_augment = image.copy()
    r             = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image_augment, cv2.COLOR_BGR2HSV))
    dtype         = image.dtype  # uint8
    x             = np.arange(0, 256, dtype=np.int16)
    
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    
    image4_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                            cv2.LUT(sat, lut_sat),
                            cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(image4_hsv, cv2.COLOR_HSV2BGR, dst=image_augment)  # no return needed
    
    # NOTE: Histogram equalization
    if equalize_hist and random.random() < 0.2:
        for i in range(3):
            image_augment[:, :, i] = cv2.equalizeHist(image_augment[:, :, i])
    return image_augment


# MARK: - Histograms
