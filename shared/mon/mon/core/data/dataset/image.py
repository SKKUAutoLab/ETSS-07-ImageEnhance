#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image-based datasets."""

__all__ = [
    "ImageLoader",
]

import glob

import albumentations as A

from mon.core.dtypes.image import Image
from mon.core.enum import Split
from mon.core.pathlib import Path
from mon.core.rich import create_progress_bar
from .base import Modalities, Modality
from .vision import VisionDataset


# ----- Image Loader -----
class ImageLoader(VisionDataset):
    """Loads images from a file path, pattern, or directory.
    
    Attributes:
        modalities: Dictionary of datapoint modalities.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.PREDICT``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root     : Path,
        split    : Split     = Split.PREDICT,
        transform: A.Compose = None,
        verbose  : bool      = True,
        *args, **kwargs
    ):
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Initialize -----
    def list_primary_data(self) -> list:
        """Retrieves ``Image`` objects from the root path.

        Raises:
            IOError: If ``root`` path invalid or no images found.
        """
        if self.root.is_image_file():
            paths = [self.root]
        elif self.root.is_dir() and self.root.exists():
            paths = list(self.root.rglob("*"))
        elif "*" in str(self.root):
            paths = [Path(i) for i in glob.glob(str(self.root))]
        else:
            raise IOError(f"Invalid root path: {self.root}")
        
        images: list[Image] = []
        with create_progress_bar() as pbar:
            paths = sorted(paths)
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(path=path, root=self.root))
        
        return images
