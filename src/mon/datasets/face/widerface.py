#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements WiderFace datasets."""

__all__ = [
    "WiderFace",
    "WiderFaceDataModule",
    "WiderFaceTest",
    "WiderFaceTestDataModule",
    "WiderFaceVal",
    "WiderFaceValDataModule",
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
@DATASETS.register(name="widerface")
class WiderFace(VisionDataset):
    """Loads WiderFace dataset from ``root`` dir.

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
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "widerface" if root.name != "widerface" else root
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


@DATASETS.register(name="widerfaceval")
class WiderFaceVal(WiderFace):
    """Loads WiderFace-Val dataset from ``root`` dir.

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


@DATASETS.register(name="widerfacetest")
class WiderFaceTest(WiderFace):
    """Loads WiderFace-Test dataset from ``root`` dir.

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
@DATAMODULES.register(name="widerface")
class WiderFaceDataModule(core.DataModule):
    """Configures WiderFace datasets for training/testing."""
    
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
            self.train = WiderFace(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = WiderFace(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = WiderFace(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="widerfaceval")
class WiderFaceValDataModule(core.DataModule):
    """Configures WiderFace-Val datasets for training/testing."""

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
            self.train = WiderFaceVal(split=Split.VAL, **self.dataset_kwargs)
            self.val   = WiderFaceVal(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = WiderFaceVal(split=Split.VAL, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="widerfacetest")
class WiderFaceTestDataModule(core.DataModule):
    """Configures WiderFace-Test datasets for training/testing."""

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
            self.train = WiderFaceVal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = WiderFaceVal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = WiderFaceVal(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
