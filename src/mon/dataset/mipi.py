#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIPI Challenges.

This module implements datasets and datamodules for MIPI challenges.

References:
	https://mipi-challenge.org/MIPI2024/index.html
"""

from __future__ import annotations

__all__ = [
	"MIPI24Flare",
	"MIPI24FlareDataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "mipi"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


# region Dataset

@DATASETS.register(name="mipi24_flare")
class MIPI24Flare(ImageDataset):
	"""Nighttime Flare Removal dataset used in MIPI 2024 Challenge.
	
	References:
		https://mipi-challenge.org/MIPI2024/index.html
	"""
	
	tasks : list[Task]  = [Task.LES]
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
	datapoint_attrs     = DatapointAttributes({
		"image"   : ImageAnnotation,
		"hq_image": ImageAnnotation,
	})
	has_test_annotations: bool = False
	
	def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
		super().__init__(root=root, *args, **kwargs)
	
	def get_data(self):
		if self.split in [Split.TRAIN]:
			patterns = [
				self.root / "train" / "mipi24_flare" / "lq",
			]
		elif self.split in [Split.VAL]:
			patterns = [
				self.root / "val" / "mipi24_flare" / "lq",
			]
		elif self.split in [Split.TEST]:
			patterns = [
				self.root / "test" / "mipi24_flare" / "lq",
			]
		else:
			raise ValueError
		
		# LQ images
		lq_images: list[ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for pattern in patterns:
				for path in pbar.track(
					sorted(list(pattern.rglob("*"))),
					description=f"Listing {self.__class__.__name__} "
					            f"{self.split_str} lq images"
				):
					if path.is_image_file():
						lq_images.append(ImageAnnotation(path=path))
		
		# HQ images
		hq_images: list[ImageAnnotation] = []
		with core.get_progress_bar(disable=self.disable_pbar) as pbar:
			for img in pbar.track(
				lq_images,
				description=f"Listing {self.__class__.__name__} "
				            f"{self.split_str} hq images"
			):
				path = img.path.replace("/lq/", "/hq/")
				hq_images.append(ImageAnnotation(path=path.image_file()))
		
		self.datapoints["image"]    = lq_images
		self.datapoints["hq_image"] = hq_images
		
		
# region DataModule

@DATAMODULES.register(name="mipi24_flare")
class MIPI24FlareDataModule(DataModule):
	"""Nighttime Flare Removal datamodule used in MIPI 2024 Challenge.
	
	References:
		https://mipi-challenge.org/MIPI2024/index.html
	"""
	
	tasks: list[Task] = [Task.LES]
	
	def prepare_data(self, *args, **kwargs):
		if self.classlabels is None:
			self.get_classlabels()
	
	def setup(self, stage: Literal["train", "test", "predict", None] = None):
		if self.can_log:
			console.log(f"Setup [red]{self.__class__.__name__}[/red].")
		
		if stage in [None, "train"]:
			self.train = MIPI24Flare(split=Split.TRAIN, **self.dataset_kwargs)
			self.val   = MIPI24Flare(split=Split.VAL, **self.dataset_kwargs)
		if stage in [None, "test"]:
			self.test  = MIPI24Flare(split=Split.VAL, **self.dataset_kwargs)
		
		if self.classlabels is None:
			self.get_classlabels()
		
		if self.can_log:
			self.summarize()
	
	def get_classlabels(self):
		pass
	
# endregion
