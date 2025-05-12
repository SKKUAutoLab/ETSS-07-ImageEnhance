#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the Cityscapes Foggy dataset.

References:
	- https://www.cityscapes-dataset.com/
"""

__all__ = [
    "CityscapesFoggy",
    "CityscapesFoggyDataModule",
]

from typing import Literal

import cv2

from mon import core, vision
from mon.constants import DATAMODULES, DATASETS, Split, Task
from mon.datasets.cityscapes.cityscapes import Cityscapes

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
InfraredAnnotation             = vision.InfraredAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="cityscapes_foggy")
class CityscapesFoggy(Cityscapes):
    """Loads and processes the CityscapesFoggy dataset for dehazing tasks.

    Args:
        root: Root directory path. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    
    Raises:
        FileNotFoundError: If ``root``/cityscapes directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
        "semantic" : SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "cityscapes" if root.name != "cityscapes" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        """Lists foggy images, reference images, and semantic maps."""
        patterns = [self.root / self.split_str / "leftImg8bit_foggy"]
        
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            for img in pbar.track(sequence=images, description=desc):
                path = img.path.replace("/leftImg8bit_foggy/", "/leftImg8bit/")
                stem = path.stem.split("leftImg8bit")[0]
                path = path.parent / f"{stem}leftImg8bit{path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} semantic maps"
            for img in pbar.track(sequence=ref_images, description=desc):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(
                    path  = path.image_file(),
                    root  = img.root,
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        self.datapoints["semantic"]  = semantic


# ----- DataModule -----
@DATAMODULES.register(name="cityscapes_foggy")
class CityscapesFoggyDataModule(core.DataModule):
    """Manages CityscapesFoggy dataset for training, validation, and testing."""

    tasks: list[Task] = [Task.DERAIN]

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
            self.train = CityscapesFoggy(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesFoggy(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesFoggy(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
