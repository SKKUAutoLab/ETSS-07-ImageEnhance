#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image-based annotations."""

__all__ = [
    "FrameAnnotation",
]

import numpy as np
import torch

from mon import core
from mon.constants import SAVE_IMAGE_EXT
from mon.vision.types import image as I


# ----- Annotation -----
class FrameAnnotation(core.Annotation):
    """Frame annotation of a video frame.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.
        
    Args:
        index: Integer index of frame in video.
        frame: Ground-truth image as ``numpy.ndarray``.
        path: Path to video file as ``core.Path`` or ``str``. Default is ``None``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        index: int,
        frame: np.ndarray,
        path : core.Path | str = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.frame = frame
        self.path  = path
        self.shape = I.image_shape(image=frame)
    
    @property
    def path(self) -> core.Path:
        """Returns the video file path.

        Returns:
            ``core.Path`` of video file or ``None`` if not set.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        """Sets the video file path.

        Args:
            path: Path to video file or ``None``.

        Raises:
            ValueError: If ``path`` is not a valid video path when provided.
        """
        if path is not None:
            path_obj = core.Path(path)
            if not path_obj.is_video_file():
                raise ValueError(f"[path] must be a valid video path, got {path}.")
            self._path = path_obj
        else:
            self._path = None

    @property
    def frame_path(self) -> core.Path:
        """Returns the path for each frame of the video: <self.path>_<self.index>.

        Returns:
            ``core.Path`` of video file or ``None`` if not set.
        """
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}{SAVE_IMAGE_EXT}"
        else:
            return self._path

    @property
    def name(self) -> str:
        """Returns the frame name.

        Returns:
            ``str`` from ``path.name`` or ``index`` if path is unset.
        """
        return self.path.name if self.path else str(self.index)
    
    @property
    def stem(self) -> str:
        """Returns the stem of the frame path.

        Returns:
            ``str`` from ``path.stem`` or ``index`` if path is unset.
        """
        return self.path.stem if self.path else str(self.index)
    
    @property
    def data(self) -> np.ndarray:
        """Returns the frame data.

        Returns:
            ``numpy.ndarray`` of frame data.
        """
        return self.frame
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the frame.

        Returns:
            Dict with ``index``, ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "index"     : self.index,
            "name"      : self.name,
            "stem"      : self.stem,
            "path"      : self.frame_path,
            "frame_path": self.frame_path,
            "video_path": self.path,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if self.path else None,
        }
    
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
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        return I.image_to_4d(batch)
