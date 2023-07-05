#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COCO 2017 dataset and datamodule.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from torchkit.core.data import ClassLabels
from torchkit.core.data import VisionData
from torchkit.core.data.image_info import ImageInfo
from torchkit.core.data.object_annotation import ObjectAnnotation
from torchkit.core.dataset import DataModule
from torchkit.core.dataset import Detection2DDataset
from torchkit.core.dataset import VisualDataHandler
from torchkit.core.fileio import is_json_file
from torchkit.core.fileio import is_txt_file
from torchkit.core.image import bbox_area
from torchkit.core.image import bbox_cxcywh_norm_xyxy
from torchkit.core.image import bbox_xyxy_cxcywh_norm
from torchkit.core.image import draw_bbox
from torchkit.core.image import is_image_file
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchkit.core.utils import Dim3
from torchkit.core.utils import LabelTypes
from torchkit.core.utils import Tasks
from torchkit.datasets.builder import DATAMODULES
from torchkit.datasets.builder import DATASETS
from torchkit.utils import datasets_dir

logger = logging.getLogger()

__all__ = ["Coco2017", "Coco2017DataModule"]


# MARK: - Data Config

data = {
	"name": "kodas2020_detection",
	# The dataset's name.
	"shape": [1536, 1536, 3],
	# The image shape as [H, W, C]. This is compatible with OpenCV format.
	# This is also used to reshape the input data.
	"num_classes": 3,
	# The number of classes in the dataset.
	"batch_size": 4,
	# Number of samples in one forward & backward pass.
	"label_format": "yolo",
	# The format to convert images and labels to when `get_item()`.
	# Each labels' format has each own directory: annotations_<format>.
	# Example: `.../train/annotations_yolo/...`
	# Supports:
	# - `custom`: uses our custom annotation format.
	# - `coco`  : uses Coco annotation format.
	# - `yolo`  : uses Yolo annotation format.
	# - `pascal`: uses Pascal annotation format.
	# Default: `yolo`.
	"caching_labels": False,
	# Should overwrite the existing cached labels? Default: `False`.
	"caching_images": False,
	# Cache images into memory for faster training. Default: `False`.
	"write_labels": False,
	# After loading images and labels for the first time, we will convert it to
	# our custom data format and write to files. If `True`, we will overwrite
	# these files. Default: `False`.
	"fast_dev_run": False,
	# Take a small subset of the data for fast debug (i.e, like unit testing).
	# Default: `False`.
	"shuffle": True,
	# Set to `True` to have the data reshuffled at every training epoch.
	# Default: `True`.
	"augment": {
		"hsv_h": 0.015,  # Image HSV-Hue augmentation (fraction).
		"hsv_s": 0.7,  # Image HSV-Saturation augmentation (fraction).
		"hsv_v": 0.4,  # Image HSV-Value augmentation (fraction).
		"rotate": 0.0,  # Image rotation (+/- deg).
		"translate": 0.5,  # Image translation (+/- fraction).
		"scale": 0.5,  # Image scale (+/- gain).
		"shear": 0.0,  # Image shear (+/- deg).
		"perspective": 0.0,  # Image perspective (+/- fraction), range 0-0.001.
		"flip_ud": 0.0,  # Image flip up-down (probability).
		"flip_lr": 0.5,  # Image flip left-right (probability).
		"mixup": 0.0,  # Image mixup (probability).
		"mosaic": True,  # Use mosaic augmentation.
		"rect": False,
		# Train model using rectangular images instead of square ones.
		"stride": 32,
		# When `rect_training=True`, reshape the image shapes as a multiply of
		# stride.
		"pad": 0.0,
		# When `rect_training=True`, pad the empty pixel with given values.
	},
}


# MARK: - Coco2017

@DATASETS.register(name="coco2017")
class Coco2017(Detection2DDataset):
	"""The COCO 2017 Dataset. This is a multi-task dataset where for each image,
	several type of ground-truth labels are associated with.

	Attributes:
		root (str):
			The dataset root directory that contains: train/val/test/...
			subdirectories.
		split (str):
			The image split to use. One of: [`train`, `val`, `test`].
			Default: `train`.
		image_paths (list):
			A list of all image filepaths.
		label_paths (list):
			A list of all label filepaths.
		data (list[VisionData]):
			The list of all `VisionData` objects.
		classlabels (ClassLabels, optional):
			The `ClassLabels` object contains all class-labels defined in the
			dataset.
		shape (tuple):
			The image shape as [H, W, C]. Used to resize the image.
		batch_size (int):
			Number of training samples in one forward & backward pass.
		batch_shapes (np.ndarray, optional):
			An array of batch shapes. It is available only for `rect_training`
			augmentation.
		batch_indexes (np.ndarray, optional):
			An array of batch indexes. It is available only for `rect_training`
			augmentation.
		label_types (LabelTypes, optional):
			COCO features two object detection tasks: using either bounding box
			output or object segmentation output (the latter is also known as
			instance segmentation). One of: [`bbox`, `polygon`]. Can be a list
			of types. Default: `bbox`.
			- `bbox`: object detection task.
			- `polygon`: instance segmentation.
		label_format (str):
			The format to convert images and labels to when `get_item()`.
			Each labels' format has each own directory:
			annotations_<format>. Example: `.../train/annotations_yolo/...`
			Supports:
				- `custom`/`default`: uses our custom annotation format.
				- `yolo`            : uses Yolo annotation format.
		label_formatter (object):
			The formatter in charge of formatting labels to the corresponding
			`label_format`.
		has_custom_labels (bool):
			Check if we have custom label files. If `True`, then those files
			will be loaded. Else, load the raw data	from the dataset, convert
			them to our custom data format, and write to files.
		caching_labels (bool):
			Should overwrite the existing cached labels?
		caching_images (bool):
			Cache images into memory for faster training.
		write_labels (bool):
			After loading images and labels for the first time, we will convert
			it to our custom data format and write to files. If `True`, we will
			overwrite these files.
		fast_dev_run (bool):
            Take a small subset of the data for fast debug (i.e, like unit
            testing).
		augment (ImageAugment):
			A data object contains all hyperparameters for augmentation
			operations.
			> Note 1: The reason for this attributes is that sometimes we want
			to use custom augmentation operations that performs on both images
			and labels.
		collate_fn (callable, None):
			The collate function used to fused input items together when using
			`batch_size > 1`. This is used in the DataLoader wrapper.
		transforms (callable, optional):
            A function/transform that takes input sample and its target as
            entry and returns a transformed version.
        transform (callable, optional):
            A function/transform that takes input sample as entry and returns
            a transformed version.
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
	"""

	# MARK: Magic Functions
	
	def __init__(
		self,
		root            : str,
		split           : str                    = "train",
		shape           : Dim3                   = (640, 640, 3),
		batch_size      : int                    = 1,
		label_types	    : Optional[LabelTypes]   = "bbox",
		label_format    : str                    = "yolo",
		caching_labels  : bool                   = False,
		caching_images  : bool                   = False,
		write_labels    : bool                   = False,
		fast_dev_run    : bool                   = False,
		augment         : Union[str, dict, None] = None,
		transforms      : Optional[Callable]   	 = None,
		transform       : Optional[Callable]   	 = None,
		target_transform: Optional[Callable]     = None,
		*args, **kwargs
	):
		self.label_types = (label_types if isinstance(label_types, list)
							else [label_types])
		
		super().__init__(
			root             = root,
			split            = split,
			shape            = shape,
			batch_size       = batch_size,
			label_format     = label_format,
			caching_labels   = caching_labels,
			caching_images   = caching_images,
			write_labels     = write_labels,
			fast_dev_run     = fast_dev_run,
			augment          = augment,
			transforms       = transforms,
			transform        = transform,
			target_transform = target_transform,
			*args, **kwargs
		)
	
	# MARK: Pre-Load Data
	
	def list_files(self):
		"""List image and label files.

		Todos:
			- Look for image and label files in `split` directory.
			- We should look for our custom label files first.
			- If none is found, proceed to listing the images and raw label
			  files.
			- Also, load `classlabels`
			- After this method, these following attributes MUST be defined:
			  `image_paths`, `label_paths`, `has_custom_labels`, `classlabels`.
		"""
		# NOTE: List image files
		image_pattern = os.path.join(
			self.root, f"2017", f"{self.split}2017", "*.jpg"
		)
		self.image_paths = glob.glob(image_pattern)
		self.image_paths = [p for p in self.image_paths
							if is_image_file(path=p)]

		# NOTE: fast_dev_run, select only a subset of images
		if self.fast_dev_run:
			indices          = [random.randint(0, len(self.image_paths) - 1)
								for _ in range(self.batch_size)]
			self.image_paths = [self.image_paths[i] for i in indices]

		# NOTE: List label files
		self.has_custom_labels = False
		self.label_paths = os.path.join(
			self.root, "2017", "annotations", f"instances_{self.split}2017.json"
		)
		
		# NOTE: Load classlabels
		classlabel_path = os.path.join(
			self.root, f"2017", f"80_classlabels.json"
		)
		if not os.path.isfile(classlabel_path):
			curr_dir        = os.path.dirname(os.path.abspath(__file__))
			classlabel_path = os.path.join(
				curr_dir, f"coco2017_80_classlabels.json"
			)
		self.classlabels = ClassLabels.create_from_file(
			label_path=classlabel_path
		)

	# MARK: Load Data
	
	def load_labels(self, image_path: str, label_path: str) -> VisionData:
		"""Load all labels from a raw label `file`.

		Args:
			image_path (str):
				The image file.
			label_path (str):
				The label file.

		Returns:
			data (VisionData):
				The `VisionData` object.
		"""
		# NOTE: If we have custom labels
		if self.has_custom_labels:
			return VisualDataHandler().load_from_file(
				image_path=image_path, label_path=label_path
			)

		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path=image_path)
		shape0     = image_info.shape0
		
		# NOTE: Parse all annotations
		with open(label_path, "r") as file_in:
			labels = [x.replace(",", " ") for x in file_in.read().splitlines()]
			labels = [x.split() for x in labels]

		# VisDrone format:
		#      0       1      2      3       4
		# <class_id> <xmin> <ymin> <xmax> <ymax>
		objs = []
		for i, l in enumerate(labels):
			class_id         = (self.classlabels.get_id(name=l[0])
								if isinstance(self.classlabels, ClassLabels)
								else l[0])
			bbox_xyxy        = np.array(l[1:], dtype=np.float32)
			bbox_cxcywh_norm = bbox_xyxy_cxcywh_norm(bbox_xyxy,
													 shape0[0], shape0[1])
			area             = bbox_area(bbox_xyxy)
			objs.append(
				ObjectAnnotation(
					class_id = class_id,
					bbox     = bbox_cxcywh_norm,
					area     = area
				)
			)
		return VisionData(image_info=image_info, objects=objs)


# MARK: - Coco2017DataModule

@DATAMODULES.register(name="coco2017")
class Coco2017DataModule(DataModule):
	"""COCO 2017 DataModule."""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dataset_dir: str = os.path.join(datasets_dir, "kodas"),
		name       : str = "kodas2020_detection2d",
		*args, **kwargs
	):
		super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
		self.dataset_kwargs = kwargs
		
	# MARK: Prepare Data
	
	def prepare_data(self, *args, **kwargs):
		"""Use this method to do things that might write to disk or that need
		to be done only from a single GPU in distributed settings.
			- Download.
			- Tokenize.
		"""
		if self.classlabels is None:
			self.load_classlabels()
	
	def setup(self, phase: Optional[Phase] = None):
		"""There are also data operations you might want to perform on every
		GPU.
		
		Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

		Args:
			phase (Phase, optional):
				The phase to use: [None, Phase.TRAINING, Phase.TESTING].
				Set to "None" to setup all train, val, and test data.
				Default: `None`.
		"""
		# NOTE: Assign train/val datasets for use in dataloaders
		if phase in [None, Phase.TRAINING]:
			self.train = Kodas2020Detection2D(
				root=self.dataset_dir, split="train", **self.dataset_kwargs
			)
			self.val = Kodas2020Detection2D(
				root=self.dataset_dir, split="val",   **self.dataset_kwargs
			)
			self.classlabels = getattr(self.train, "classlabels", None)
			self.collate_fn  = getattr(self.train, "collate_fn",  None)

		# NOTE: Assign test datasets for use in dataloader(s)
		if phase in [None, Phase.TESTING]:
			self.test = Kodas2020Detection2D(
				root=self.dataset_dir, split="val", **self.dataset_kwargs
			)
			self.classlabels = getattr(self.test, "classlabels", None)
			self.collate_fn  = getattr(self.test, "collate_fn",  None)
		
		if self.classlabels is None:
			self.load_classlabels()
	
	def load_classlabels(self):
		current_dir 	= os.path.dirname(os.path.abspath(__file__))
		classlabel_path = os.path.join(
			current_dir, f"kodas2020_classlabels.json"
		)
		self.classlabels = ClassLabels.create_from_file(
			label_path=classlabel_path
		)


# MARK: - Main

if __name__ == "__main__":
	# NOTE: Get DataModule
	cfgs = data
	dm   = Kodas2020Detection2DDataModule(**cfgs)
	dm.setup()
	# NOTE: Visualize labels
	print(f"Classlabels \n{dm.classlabels.list}")
	# NOTE: Visualize an iteration
	data_iter             = iter(dm.train_dataloader)
	images, labels, shape = next(data_iter)

	drawings = []
	for i, img in enumerate(images):
		chw       = img.shape
		l         = labels[labels[:, 0] == i]
		l[:, 2:6] = bbox_cxcywh_norm_xyxy(l[:, 2:6], chw[1], chw[2])
		drawing   = draw_bbox(img, l, dm.classlabels.colors(), 5)
		drawings.append(drawing)
	
	show_images(images=drawings, nrow=1)
	plt.show(block=True)
