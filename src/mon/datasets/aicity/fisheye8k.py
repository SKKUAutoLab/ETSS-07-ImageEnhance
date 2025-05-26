#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FishEye8K datasets."""

__all__ = [
    "FishEye8K",
    "FishEye8KDataModule",
    "FishEye8KTest",
    "FishEye8KTestDataModule",
    "FishEye8KVal",
    "FishEye8KValDataModule",
]

from typing import Literal

from mon import core, vision
from mon.constants import DATAMODULES, DATASETS, Split, Task

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
InfraredAnnotation             = vision.InfraredAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="fisheye8k")
class FishEye8K(VisionDataset):
    """Loads FishEye8K dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DETECT]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        # "bbox" : BBoxesAnnotation,
    })
    has_test_annotations: bool = False
    classlabels         = ClassLabels([
        {"name": "bus",        "id": 0, "color": [140,  24, 143]},
        {"name": "bike",       "id": 1, "color": [122,  35,   2]},
        {"name": "car",        "id": 2, "color": [ 49,   3, 150]},
        {"name": "pedestrian", "id": 3, "color": [ 81, 120, 228]},
        {"name": "truck",      "id": 4, "color": [ 72, 153, 152]},
    ])

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fisheye8k" if root.name != "fisheye8k" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="fisheye8kval")
class FishEye8KVal(FishEye8K):
    """Loads FishEye8K-Val dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "val" / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="fisheye8ktest")
class FishEye8KTest(FishEye8K):
    """Loads FishEye8K-Test dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "test" / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


# ----- DataModule -----
@DATAMODULES.register(name="fisheye8k")
class FishEye8KDataModule(core.DataModule):
    """Configures FishEye8K datasets for training/testing."""
    
    tasks: list[Task] = [Task.DETECT]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = FishEye8K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FishEye8K(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FishEye8K(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fisheye8kval")
class FishEye8KValDataModule(core.DataModule):
    """Configures FishEye8K-Val datasets for training/testing."""

    tasks: list[Task] = [Task.DETECT]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FishEye8KVal(split=Split.VAL, **self.dataset_kwargs)
            self.val   = FishEye8KVal(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FishEye8KVal(split=Split.VAL, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fisheye8ktest")
class FishEye8KTestDataModule(core.DataModule):
    """Configures FishEye8K-Test datasets for training/testing."""

    tasks: list[Task] = [Task.DETECT]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FishEye8KVal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = FishEye8KVal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FishEye8KVal(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
