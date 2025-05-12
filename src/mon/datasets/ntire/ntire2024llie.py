#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements datasets and datamodules for NTIRE 2024 LLIE Challenge.

References:
    - https://codalab.lisn.upsaclay.fr/competitions/17640
"""

__all__ = [
    "NTIRE2024LLIE",
    "NTIRE2024LLIEDataModule",
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
@DATASETS.register(name="ntire2024llie")
class NTIRE2024LLIE(VisionDataset):
    """Loads NTIRE 2024 LLIE dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    tasks : list[Task]  =  [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "ntire2024llie" if root.name != "ntire2024llie" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split.

        Raises:
            ValueError: If ``split`` is invalid.
        """
        if self.split in [Split.TRAIN]:
            patterns = [self.root / "train" / "image"]
        elif self.split in [Split.VAL]:
            patterns = [self.root / "val" / "image"]
        elif self.split in [Split.TEST]:
            patterns = [self.root / "test" / "image"]
        else:
            raise ValueError(f"[split] invalid: [{self.split}]")

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
@DATAMODULES.register(name="ntire2024llie")
class NTIRE2024LLIEDataModule(core.DataModule):
    """Configures NTIRE 2024 LLIE datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data for disk or single-GPU tasks."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red]")
        
        if stage in [None, "train"]:
            self.train = NTIRE2024LLIE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = NTIRE2024LLIE(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = NTIRE2024LLIE(split=Split.VAL,   **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
