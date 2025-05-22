#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements depth-based annotations."""

__all__ = [
    "DepthMapAnnotation",
]

from typing import Literal

import cv2

from mon import core
from mon.constants import DepthDataSource
from mon.vision.types import image as I


# ----- Annotation -----
class DepthMapAnnotation(I.ImageAnnotation):
    """Dense depth map annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.
    
    Args:
        path: Path to depth map file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``'dav2_vitb'``.
        source: Source of depth data from ``DepthDataSource``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_GRAYSCALE``).
            Default is ``cv2.IMREAD_GRAYSCALE``.

    Raises:
        ValueError: If ``source`` is not in ``DEPTH_DATA_SOURCES``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        path  : core.Path | str,
        root  : core.Path | str = None,
        source: Literal[*DepthDataSource.values()] = "dav2_vitl",
        flags : int = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        if source not in DepthDataSource:
            raise ValueError(f"[source] must be one of {DepthDataSource}, got {source}.")
        self.source = source
        self.flags  = (cv2.IMREAD_COLOR if source and "c" in source else cv2.IMREAD_GRAYSCALE)
