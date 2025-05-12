#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GT-Rain datasets."""

__all__ = [
    "GTRain",
    "GTRainDataModule",
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
@DATASETS.register(name="gtrain")
class GTRain(VisionDataset):
    """Loads GTRain dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "gtrain" if root.name != "gtrain" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image and ref annotations."""
        patterns = [self.root / self.split_str / "image"]

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
                path = str(img.path)
                if "Gurutto_1-2" in path:
                    path = path.replace("-R-", "-C-")
                else:
                    path = path[:-9] + "C-000.png"
                path = path.replace("/image/", "/ref/")
                path = core.Path(path)
                ref_images.append(ImageAnnotation(path=path.image_file()))

        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images

        
# ----- DataModule -----
@DATAMODULES.register(name="gtrain")
class GTRainDataModule(core.DataModule):
    """Configures GTRain datasets for training/testing."""
    
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
            self.train = GTRain(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTRain(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTRain(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
