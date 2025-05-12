#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SICE datasets."""

__all__ = [
    "SICE",
    "SICEDataModule",
    "SICEGrad",
    "SICEGradDataModule",
    "SICEVE",
    "SICEMEDataModule",
    "SICEMix",
    "SICEMixDataModule",
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
@DATASETS.register(name="sice")
class SICE(VisionDataset):
    """Loads SICE dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "sice" if root.name != "sice" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / "sice" / self.split_str / "image"]

        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="sicegrad")
class SICEGrad(VisionDataset):
    """Loads SICE-Grad dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "sice" if root.name != "sice" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / "grad" / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="sicemix")
class SICEMix(VisionDataset):
    """Loads SICE-Mix dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "sice" if root.name != "sice" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / "mix" / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="siceve")
class SICEVE(VisionDataset):
    """Loads SICE-VE dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "sice" if root.name != "sice" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / "ve" / self.split_str / "image"]
        
        # Images
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
@DATAMODULES.register(name="sice")
class SICEDataModule(core.DataModule):
    """Configures SICE datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for a specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = SICE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICE(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICE(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="sicegrad")
class SICEGradDataModule(core.DataModule):
    """Configures SICE-Grad datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for a specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = SICEGrad(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEGrad(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEGrad(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="sicemix")
class SICEMixDataModule(core.DataModule):
    """Configures SICE-Mix datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for a specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = SICEMix(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEMix(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEMix(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
            
            
@DATAMODULES.register(name="siceve")
class SICEMEDataModule(core.DataModule):
    """Configures SICE-VE datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for a specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = SICEVE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEVE(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEVE(split=Split.TEST, **self.dataset_kwargs)
            
        self.get_classlabels()
        if self.can_log:
            self.summarize()
