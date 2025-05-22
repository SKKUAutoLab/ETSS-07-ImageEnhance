#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ExDark datasets.

References:
    - https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
"""

__all__ = [
    "ExDark",
    "ExDark1200",
    "ExDark1200DataModule",
    "ExDarkDataModule",
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
@DATASETS.register(name="exdark")
class ExDark(VisionDataset):
    """Loads ExDark dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE, Task.DETECT]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    classlabels         = ClassLabels([
        {"name": "Bicycle"  , "id":  1, "coco80_id":  2, "color": [138, 183,  33]},
        {"name": "Boat"     , "id":  2, "coco80_id":  9, "color": [ 19,  64,  83]},
        {"name": "Bottle"   , "id":  3, "coco80_id": 40, "color": [139, 160,   1]},
        {"name": "Bus"      , "id":  4, "coco80_id":  6, "color": [140,  24, 143]},
        {"name": "Car"      , "id":  5, "coco80_id":  3, "color": [ 49,   3, 150]},
        {"name": "Cat"      , "id":  6, "coco80_id": 16, "color": [ 41, 174, 251]},
        {"name": "Chair"    , "id":  7, "coco80_id": 57, "color": [ 94, 173,  36]},
        {"name": "Cup"      , "id":  8, "coco80_id": 42, "color": [ 28,  47,  55]},
        {"name": "Dog"      , "id":  9, "coco80_id": 17, "color": [ 21,   8, 251]},
        {"name": "Motorbike", "id": 10, "coco80_id":  4, "color": [122,  35,   2]},
        {"name": "People"   , "id": 11, "coco80_id":  1, "color": [ 81, 120, 228]},
        {"name": "Table"    , "id": 12, "coco80_id": 61, "color": [216, 147, 179]},
    ])

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "exdark" if root.name != "exdark" else root
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


@DATASETS.register(name="exdark1200")
class ExDark1200(ExDark):
    """A subset of ExDark dataset with 1200 images (first 100 images from each subset).

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / f"{self.split_str}1200" / "image"]

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
@DATAMODULES.register(name="exdark")
class ExDarkDataModule(core.DataModule):
    """Configures ExDark datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
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
            self.train = ExDark(split=Split.TEST, **self.dataset_kwargs)
            self.val   = ExDark(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = ExDark(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="exdark1200")
class ExDark1200DataModule(core.DataModule):
    """Configures ExDarkFull datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
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
            self.train = ExDark1200(split=Split.TEST, **self.dataset_kwargs)
            self.val   = ExDark1200(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = ExDark1200(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
