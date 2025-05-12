#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements thermal and infrared annotations."""

__all__ = [
    "InfraredAnnotation",
]

from typing import Literal

import cv2

from mon import core
from mon.constants import InfraredDataSource
from mon.vision.types import image as I


# ----- Annotation -----
class InfraredAnnotation(I.ImageAnnotation):
    """Dense infrared map annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.
    
    Args:
        path: Path to infrared map file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``'infrared'``.
        source: Source of depth data from ``InfraredDataSource``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_GRAYSCALE``).
            Default is ``cv2.IMREAD_GRAYSCALE``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        path  : core.Path | str,
        root  : core.Path | str = None,
        source: Literal[*InfraredDataSource.values()] = "infrared",
        flags : int = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        if source not in InfraredDataSource:
            raise ValueError(f"[source] must be one of {InfraredDataSource}, got {source}.")
        self.source = source
