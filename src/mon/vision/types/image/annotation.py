#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image-based annotations."""

__all__ = [
    "ImageAnnotation",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.vision.types.image import io, processing


# ----- Annotation -----
class ImageAnnotation(core.Annotation):
    """Image annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        path: Path to image file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``).
            Default is ``cv2.IMREAD_COLOR``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        path : core.Path | str,
        root : core.Path | str = None,
        flags: int = cv2.IMREAD_COLOR_BGR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root   = root
        self.path   = path
        self.flags  = flags
        self.image  = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        """Returns the image file path.

        Returns:
            ``core.Path`` of the image file path.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str):
        """Sets the image file path.

        Args:
            path: Path to image file as ``core.Path`` or ``str``.

        Raises:
            ValueError: If ``path`` is not a valid image path.
        """
        path_obj = core.Path(path)
        if not path or not path_obj.is_image_file():
            raise ValueError(f"[path] must be a valid image path, got {path}.")
        self._path  = path_obj
        self._shape = io.read_image_shape(path=self._path)
    
    @property
    def name(self) -> str:
        """Returns the image file name.

        Returns:
            ``str`` of the image file name.
        """
        return self.path.name
    
    @property
    def stem(self) -> str:
        """Returns the stem of the image file path.

        Returns:
            ``str`` of the image file path stem.
        """
        return self.path.stem
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the image shape.

        Returns:
            Tuple of [H, W, C] for image dimensions.
        """
        return self._shape
    
    @property
    def data(self) -> np.ndarray | None:
        """Returns the image data.

        Returns:
            ``numpy.ndarray`` of image data or ``None`` if not loaded.
        """
        return self.image if self.image is not None else self.load(flags=self.flags, cache=False)
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }
    
    def load(
        self,
        path : core.Path | str = None,
        flags: int  = None,
        cache: bool = False
    ) -> np.ndarray:
        """Loads the image into memory.

        Args:
            path: Path to image file. Default is ``None``.
            flags: Flag to read image. Default is ``None``.
            cache: If ``True``, caches image. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        # Return the image if it is already loaded
        if self.image is not None:
            return self.image
        # Load the image
        load_path  = path  or self.path
        load_flags = flags or self.flags
        image      = io.load_image(load_path, load_flags, False, False)
        # Update the shape of the image
        if self._shape != image.shape:
            self._shape = image.shape
        # Cache the image if needed
        self.image = image if cache else None
        if path:
            self.path  = load_path
        if flags:
            self.flags = load_flags
        return image
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.
            normalize: If ``True``, normalizes data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        return processing.image_to_tensor(data, normalize)
    
    @staticmethod
    def collate_fn(batch: list) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        return processing.image_to_4d(batch)
