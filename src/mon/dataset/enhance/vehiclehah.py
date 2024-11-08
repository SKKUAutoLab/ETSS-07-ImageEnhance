#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VehicleHaH Datasets."""

from __future__ import annotations

__all__ = [
    "VehicleHaH",
    "DICMDataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="dicm")
class VehicleHaH(MultimodalDataset):
    """Data from vehicles driving on snowy nights reveal a range of challenges
    and trials. When the vehicle is driving on the snow, it needs a longer
    braking distance and is easy to lose control. The vehicle's powertrain may
    be affected by the slippery road surface, which reduces traction.
    Visibility may be reduced by snow and wind, making it difficult for drivers
    to see the road and other vehicles. Therefore, when driving on a snowy
    night, the driver needs to be extra careful, appropriately reduce the speed,
    maintain a safe distance, and ensure that the vehicle is in good working
    condition. At the same time, collecting data on the driving of these
    vehicles on snowy nights can help improve vehicle design and driving
    techniques to improve driving safety.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "dicm" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATAMODULES.register(name="dicm")
class DICMDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = VehicleHaH(split=Split.TEST, **self.dataset_kwargs)
            self.val   = VehicleHaH(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = VehicleHaH(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
