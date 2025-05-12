#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements label map annotations."""

__all__ = [
    "SemanticSegmentationAnnotation",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.vision.types import image as I


# ----- Annotation -----
class SemanticSegmentationAnnotation(core.Annotation):
    """Semantic segmentation annotation (mask).
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``mask``.

    Args:
        path: Path to image file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR_BGR``).
            Default is ``cv2.IMREAD_COLOR_BGR``.
    """
    
    albumentation_target_type: str = "mask"
    
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
        self.mask   = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        """Returns the image file path.

        Returns:
            ``core.Path`` of the image file path.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        """Sets the image file path.

        Args:
            path: Path to image file or ``None``.

        Raises:
            ValueError: If ``path`` is not a valid image path or is ``None``.
        """
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f"[path] must be a valid image path, got {path}.")
        self._path  = core.Path(path)
        self._shape = I.read_image_shape(path=self._path)
    
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
        """Returns the mask shape.

        Returns:
            Tuple of [H, W, C] for mask dimensions.
        """
        return self._shape
    
    @property
    def data(self) -> np.ndarray | None:
        """Returns the mask data.

        Returns:
            ``numpy.ndarray`` of mask data or ``None`` if not loaded.
        """
        return self.mask if self.mask is not None else self.load()
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the mask.

        Returns:
            Dict with ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
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
        cache: bool = False,
    ) -> np.ndarray | None:
        """Loads the mask into memory.

        Args:
            path: Path to image file. Default is ``None``.
            flags: Flag to read image. Default is ``None``.
            cache: If ``True``, caches mask. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C], values in [0, 255], or ``None``.
        """
        if self.mask is not None:
            return self.mask
        load_path  = path  or self.path
        load_flags = flags or self.flags
        mask       = I.load_image(load_path, load_flags, False, False)
        self.mask = mask if cache else None
        return mask
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.
            normalize: If ``True``, normalizes data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        return I.image_to_tensor(data, normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of masks as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        return I.image_to_4d(batch)
