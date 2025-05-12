#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Flare7K datasets."""

__all__ = [
    "Flare7KPPExtra",
    "Flare7KPPExtraDataModule",
    "Flare7KPPReal",
    "Flare7KPPRealDataModule",
    "Flare7KPPSynthetic",
    "Flare7KPPSyntheticDataModule",
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
@DATASETS.register(name="flare7k++_real")
class Flare7KPPReal(VisionDataset):
    """Loads Flare7K++Real dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "flare7k++" if root.name != "flare7k++" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "real" / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

@DATASETS.register(name="flare7k++_synthetic")
class Flare7KPPSynthetic(VisionDataset):
    """Loads Flare7K++Synthetic dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "flare7k++" if root.name != "flare7k++" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "synthetic" / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="flare7k++_extra")
class Flare7KPPExtra(VisionDataset):
    """Loads Flare7K++Extra dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "flare7k++" if root.name != "flare7k++" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "extra" / "image"]

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
@DATAMODULES.register(name="flare7k++_real")
class Flare7KPPRealDataModule(core.DataModule):
    """Configures Flare7KPPReal datasets for training/testing."""
    
    tasks: list[Task] = [Task.NIGHTTIME]

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
            self.train = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
    

@DATAMODULES.register(name="flare7k++_synthetic")
class Flare7KPPSyntheticDataModule(core.DataModule):
    """Configures Flare7KPPSynthetic datasets for training/testing."""
    
    tasks: list[Task] = [Task.NIGHTTIME]

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
            self.train = Flare7KPPSynthetic(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPSynthetic(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPSynthetic(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="flare7k++_extra")
class Flare7KPPExtraDataModule(core.DataModule):
    """Configures Flare7KPPExtra datasets for training/testing."""
    
    tasks: list[Task] = [Task.NIGHTTIME]

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
            self.train = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
