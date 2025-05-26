#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements templates for video-only datasets."""

__all__ = [
    "VideoLoader",
    "VideoLoaderCV",
    "is_video_dataset",
]

from abc import ABC
from typing import Any

import cv2

from mon import core
from mon.constants import Split, Task, TRANSFORMS, SAVE_IMAGE_EXT
from mon.vision.geometry import albumentation
from mon.vision.types.video import FrameAnnotation


# ----- Video Loader -----
class VideoLoader(core.Dataset, ABC):
    """Base class for video loaders.

    Attributes:
        datapoint_attrs: Dict of attribute names and types.

    Args:
        root: Path to video file or stream.
        split: Data split to use. Default is ``Split.PREDICT``.
        transform: Transformations for input/target. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    tasks: list[Task] = [Task.VIDEO]
    datapoint_attrs   = core.DatapointAttributes({
        "image": FrameAnnotation,
    })
    
    def __init__(
        self,
        root      : core.Path,
        split     : Split = Split.PREDICT,
        transform : albumentation.Compose = None,
        to_tensor : bool = False,
        cache_data: bool = False,
        verbose   : bool = True,
        *args, **kwargs
    ):
        self.num_frames = 0
        super().__init__(
            root        = root,
            split       = split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    # ----- Magic Methods -----
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at given ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            ``dict`` with datapoint and metadata.
        """
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            main_attr      = self.main_attribute
            args           = {k: v for k, v in datapoint.items() if v}
            args["image"]  = args.pop(main_attr)
            transformed    = self.transform(**args)
            transformed[main_attr] = transformed.pop("image")
            datapoint     |= transformed
        
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = getattr(self.datapoint_attrs[k], "to_tensor", None)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, normalize=True)
        
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Gets total number of frames.

        Returns:
            Number of frames in video.
        """
        return self.num_frames
    
    # ----- Properties -----
    @property
    def albumentation_target_types(self) -> dict[str, str]:
        """Gets the target types for Albumentations.
        
        Returns:
            ``dict`` with keys as attribute names and values as target types.
        """
        target_types = {}
        for k, v in self.datapoint_attrs.items():
            target_type = getattr(v, "albumentation_target_type", None)
            if target_type:
                target_types[k] = target_type
        
        target_types.pop("meta", None)
        return target_types
    
    # ----- Initialize -----
    def init_transform(self, transform: albumentation.Compose | Any = None):
        """Initializes transformations with multimodal support.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        if transform is None:
            self.transform = None
        elif isinstance(transform, albumentation.Compose):
            self.transform = transform
        else:
            transform  = [transform] if not isinstance(transform, (list, tuple)) else transform
            transform_ = []
            for t in transform:
                if isinstance(t, albumentation.BasicTransform):
                    transform_.append(t)
                elif isinstance(t, dict):
                    t_ = TRANSFORMS.build(config=t)
                    if t_:
                        transform_.append(t_)
                    else:
                        raise ValueError(f"Transform [{t}] is not supported.")
                else:
                    raise TypeError(f"[transform] must be a list of albumentation.BasicTransform "
                                    f"or dicts, got {type(t)}.")
            self.transform = albumentation.Compose(transforms=transform_)
            
        if isinstance(self.transform, albumentation.Compose):
            additional_targets = self.albumentation_target_types
            additional_targets.pop(self.main_attribute, None)
            self.transform.add_targets(additional_targets)
    
    def filter_data(self):
        """Filters unwanted datapoints."""
        pass
    
    def verify_data(self):
        """Verifies dataset integrity.

        Raises:
            RuntimeError: If no datapoints exist.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        if self.verbose:
            core.console.log(f"Number of {self.split_str} datapoints: {self.__len__()}.")


class VideoLoaderCV(VideoLoader):
    """Loads video frames from a file or stream using ``cv2``.

    Args:
        root: Path to video file or stream.
        split: Data split to use. Default is ``Split.PREDICT``.
        transform: Transformations to apply. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    def __init__(
        self,
        root      : core.Path,
        split     : Split = Split.PREDICT,
        transform : albumentation.Compose = None,
        to_tensor : bool = False,
        cache_data: bool = False,
        verbose   : bool = True,
        *args, **kwargs
    ):
        self.video_capture = None
        super().__init__(
            root        = root,
            split       = split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    # ----- Properties -----
    @property
    def is_stream(self) -> bool:
        """Checks if input is a video stream.

        Returns:
            ``True`` if input is stream, ``False`` otherwise.
        """
        return self.root.is_video_stream() or self.num_frames == -1
    
    @property
    def format(self):
        """Gets format of Mat objects.

        Returns:
            Format code from ``VideoCapture.retrieve()``; -1 for RAW streams.
        """
        return self.video_capture.get(cv2.CAP_PROP_FORMAT)
    
    @property
    def fourcc(self) -> str:
        """Gets 4-character codec code.

        Returns:
            FourCC code as string.
        """
        return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
    
    @property
    def fps(self) -> int:
        """Gets frame rate.

        Returns:
            Frames per second as integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))
    
    @property
    def frame_height(self) -> int:
        """Gets height of video frames.

        Returns:
            Frame height in pixels as integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def frame_width(self) -> int:
        """Gets width of video frames.

        Returns:
            Frame width in pixels as integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Gets shape of video frames.

        Returns:
            Tuple of (height, width, channels) as integers.
        """
        return self.frame_height, self.frame_width, 3
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Gets image size of video frames.

        Returns:
            Tuple of (height, width) as integers.
        """
        return self.frame_height, self.frame_width
    
    @property
    def mode(self):
        """Gets current capture mode.

        Returns:
            Backend-specific mode value.
        """
        return self.video_capture.get(cv2.CAP_PROP_MODE)
    
    @property
    def pos_avi_ratio(self) -> int:
        """Gets relative position in video.

        Returns:
            Integer from ``0`` (start) to ``1`` (end).
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    
    @property
    def pos_msec(self) -> int:
        """Gets current position in milliseconds.

        Returns:
            Position in milliseconds as integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
    
    @property
    def pos_frames(self) -> int:
        """Gets next frame index.

        Returns:
            0-based index of next frame as integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    # ----- Initialize -----
    def list_data(self):
        """Gets video data from root path.

        Raises:
            IOError: If root not a valid video file or stream.
        """
        root = core.Path(self.root)
        if root.is_video_file():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif root.is_video_stream():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = -1
        else:
            raise IOError(f"Invalid video source: {self.root}")
        
        if self.num_frames != num_frames:
            self.num_frames = num_frames
    
    def reset(self):
        """Resets the video loader."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Closes and releases video capture."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.release()
    
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict containing datapoint data.

        Raises:
            StopIteration: If index exceeds frame count for non-streams.
            RuntimeError: If ``video_capture`` not initialized.
        """
        if not self.is_stream and index >= self.num_frames:
            self.close()
            raise StopIteration
        
        if isinstance(self.video_capture, cv2.VideoCapture):
            ret_val, frame = self.video_capture.read()
        else:
            raise RuntimeError("[video_capture] has not been initialized.")
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = FrameAnnotation(index=index, frame=frame, path=self.root)

        datapoint = self.new_datapoint
        for k, v in self.datapoints.items():
            if k == self.main_attribute:
                datapoint[k] = frame.data
            elif v is not None and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data

        return datapoint
    
    def get_meta(self, index: int = 0) -> dict:
        """Gets metadata at specified index.

        Args:
            index: Index of metadata. Default is ``0``.

        Returns:
            Dict with metadata from main attribute.
        """
        path = self.root
        return {
            "format"       : self.format,
            "fourcc"       : self.fourcc,
            "fps"          : self.fps,
            "frame_height" : self.frame_height,
            "frame_width"  : self.frame_width,
            "hash"         : self.root.stat().st_size if isinstance(self.root, core.Path) else None,
            "image_size"   : (self.frame_height, self.frame_width),
            "imgsz"        : (self.frame_height, self.frame_width),
            "index"        : index,
            "mode"         : self.mode,
            "name"         : str(self.root.name),
            "num_frames"   : self.num_frames,
            "path"         : path.parent / path.stem / f"{path.stem}_{index}{SAVE_IMAGE_EXT}",
            "frame_path"   : path.parent / path.stem / f"{path.stem}_{index}{SAVE_IMAGE_EXT}",
            "video_path"   : path,
            "pos_avi_ratio": self.pos_avi_ratio,
            "pos_frames"   : self.pos_frames,
            "pos_msec"     : self.pos_msec,
            "shape"        : (self.frame_height, self.frame_width, 3),
            "split"        : self.split_str,
            "stem"         : str(self.root.stem),
        }


# ----- Validation Check -----
def is_video_dataset(dataset: core.Dataset) -> bool:
    """Checks if dataset is a video dataset.

    Args:
        dataset: Dataset to check.

    Returns:
        ``True`` if dataset is a video dataset, ``False`` otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, (list, tuple)):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, (VideoLoader, VideoLoaderCV))
