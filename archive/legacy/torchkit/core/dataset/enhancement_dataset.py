#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The template for datasets used in image enhancement task.
"""

from __future__ import annotations

import logging
import os
import random
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from torchkit.core.data import ClassLabels
from ..data.image_info import ImageInfo
from torchkit.core.data import ImageAugment
from torchkit.core.data import VisionData
from torchkit.core.fileio import create_dirs
from torchkit.core.fileio import get_hash
from torchkit.core.image import random_perspective_mask
from torchkit.core.image import resize_image
from torchkit.core.utils import Dim3
from .formatter import LABEL_FORMATTERS
from .handler import VisualDataHandler

logger = logging.getLogger()

__all__ = ["EnhancementDataset"]


# MARK: - EnhancementDataset

class EnhancementDataset(VisionDataset, metaclass=ABCMeta):
	"""Enhancement dataset is a base class that defines the pattern for all
	image enhancement datasets.

	Attributes:
		root (str):
			The dataset root directory that contains: train/val/test/...
			subdirectories.
		split (str):
			The split to use. One of: ["train", "val", "test"].
		image_paths (list):
			A list of all image filepaths.
		eimage_paths (list):
			A list of all enhanced image filepaths.
		label_paths (list):
			A list of all label filepaths that can provide extra info about the
			image and enhanced image.
		data (list):
			The list of all `VisionData` objects.
		classlabels (ClassLabels, optional):
			The `ClassLabels` object contains all class-labels defined in the
			dataset. Default: `None`.
		shape (tuple):
			The image shape as [H, W, C] Use to resize the image.
		label_formatter (object):
			The formatter in charge of formatting labels to the corresponding
			`label_format`.
		has_custom_labels (bool):
			Check if we have custom label files. If `True`, then those files
			will be loaded. Else, load the raw data from the dataset, convert
			them to our custom data format, and write to files.
		caching_labels (bool):
			Should overwrite the existing cached labels?
		caching_images (bool):
			Cache images into memory for faster training..
		write_labels (bool):
			After loading images and labels for the first time, we will convert
			it to our custom data format and write to files.
		fast_dev_run (bool):
			Take a small subset of the data for fast debug (i.e, like unit
			testing).
		augment (ImageAugment):
			A data object contains all hyperparameters for augmentation
			operations.
			> Note 1: The reason for this attributes is that sometimes we want
			to use custom augmentation operations that performs on both
			images and labels.
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
		split           : str,
		classlabels     : ClassLabels            = None,
		shape           : Dim3                   = (720, 1280, 3),
		caching_labels  : bool                   = False,
		caching_images  : bool                   = False,
		write_labels    : bool                   = False,
		fast_dev_run    : bool                   = False,
		augment         : Union[str, dict, None] = None,
		transforms      : Optional[Callable]     = None,
		transform       : Optional[Callable]     = None,
		target_transform: Optional[Callable]     = None,
		*args, **kwargs
	):
		super().__init__(
			root             = root,
			transforms       = transforms,
			transform        = transform,
			target_transform = target_transform
		)
		self.split             = split
		self.image_paths       = []
		self.eimage_paths      = []
		self.label_paths       = []
		self.data              = []
		self.classlabels       = classlabels
		self.shape             = shape
		self.has_custom_labels = False
		self.caching_labels    = caching_labels
		self.caching_images    = caching_images
		self.write_labels      = write_labels
		self.fast_dev_run      = fast_dev_run
		
		# NOTE: Define augmentation parameters
		if isinstance(augment, dict):
			self.augment = ImageAugment(**augment)
		elif isinstance(augment, str):
			self.augment = ImageAugment().from_file(path=augment)
		else:
			self.augment = ImageAugment()
			
		# NOTE: Pre-load data. List all image files and label files
		self.list_files()
		# NOTE: Load data. Load and cache images and labels
		self.load_data()
		# NOTE: Post-load data
		self.post_load_data()
		# NOTE: Define label_formatter
		self.label_formatter = LABEL_FORMATTERS.build(name="visual",
													  dataset=self)
		self.collate_fn = getattr(self.label_formatter,
								  "collate_enhancement_fn", None)
		
		# NOTE: Define transforms
		if self.transform is None:
			self.transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()
			])
		if self.target_transform is None:
			self.target_transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()
			])
	
	def __len__(self) -> int:
		"""Return the size of the dataset."""
		return len(self.image_paths)
	
	def __getitem__(self, index: int) -> Any:
		"""Return a tuple of data item from the dataset. Depend on the
		`label_format`, we will need to convert the data.
		"""
		items  = self.label_formatter.get_enhancement_item(index=index)
		image  = items[0]
		target = items[1]
		rest   = items[2:]
		
		if self.transform is not None:
			image  = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.transforms is not None:
			image  = self.transforms(image)
			target = self.transforms(target)
		return image, target, rest
	
	# MARK: Pre-Load Data
	
	@abstractmethod
	def list_files(self):
		"""List image and label files.

		Todos:
			- Look for image and label files in `split` directory.
			- We should look for our custom label files first.
			- If none is found, proceed to listing the images and raw labels'
			  files.
			- After this method, these following attributes MUST be defined:
			  `image_paths`, `eimage_paths`, `label_paths`, `has_custom_labels`.
		"""
		pass
	
	# MARK: Load Data
	
	def load_data(self):
		"""Load and cache images, enhanced images, and labels.
		"""
		# NOTE: Check labels cache
		path              = self.eimage_paths[0]
		split_prefix      = path[ : path.find(self.split)]
		cached_label_path = f"{split_prefix}{self.split}.cache"
		
		if os.path.isfile(cached_label_path):
			cache = torch.load(cached_label_path)  # Load
			if self.caching_labels:  # Force re-cache
				cache = self.cache_labels(path=cached_label_path)  # Re-cache
			elif cache["hash"] != get_hash(self.label_paths +
										   self.image_paths +
										   self.eimage_paths):  # Dataset changed
				cache = self.cache_labels(path=cached_label_path)  # Re-cache
		else:
			cache = self.cache_labels(path=cached_label_path)  # Cache
	
		# NOTE: Get labels
		self.data = [cache[x] for x in self.image_paths]

		# NOTE: Cache images
		if self.caching_images:
			self.cache_images()
			self.cache_enhanced_images()
	
	def cache_labels(self, path: str) -> dict:
		"""Cache labels, check images and read shapes.

		Args:
			path (str):
				The path to save the cached labels.

		Returns:
			x (dict):
				The dictionary contains the labels (numpy array) and the
				original image shapes that were cached.
		"""
		# NOTE: Load all labels in label files
		cache_labels = {}
		has_label    = len(self.label_paths) == len(self.image_paths)
		for i in tqdm(range(len(self.image_paths)),
					  desc=f"Caching {self.split} labels"):
			# NOTE: Get labels
			label_path = (self.label_paths[i]
				if (has_label and os.path.isfile(self.label_paths[i]))
				else None)
			labels = self.load_labels(
				image_path  = self.image_paths[i],
				eimage_path = self.eimage_paths[i],
				label_path  = label_path
			)
		
			# NOTE: Add everything to dictionary
			cache_labels[self.image_paths[i]] = labels
			
			# NOTE: Check for any changes btw the cached labels
			if labels.image_info.path != self.image_paths[i]:
				self.caching_labels = True
		
		# NOTE: Write cache
		logger.info(f"Labels has been cached to: {path}.")
		cache_labels["hash"] = get_hash(self.label_paths +
										self.image_paths +
										self.eimage_paths)
		torch.save(cache_labels, path)  # Save for next time
		return cache_labels
	
	@abstractmethod
	def load_labels(
		self,
		image_path : str,
		eimage_path: str,
		label_path : Optional[str] = None
	) -> VisionData:
		"""Load all labels from a raw label `file`.

		Args:
			image_path (str):
				The image filepath.
			eimage_path (str):
				The enhanced image filepath.
			label_path (str, optional):
				The label filepath. Default: `None`.

		Returns:
			data (VisionData):
				The `VisionData` object.
		"""
		pass
	
	def cache_images(self):
		"""Cache images into memory for faster training (WARNING: large
		datasets may exceed system RAM).
		"""
		n    = len(self.image_paths)
		gb   = 0  # Gigabytes of cached images
		pbar = tqdm(range(n), desc="Caching images")
		for i in pbar:  # Should be max 10k images
			# image, hw_original, hw_resized
			(self.data[i].image,
			 self.data[i].image_info) = self.load_image(index=i)
			gb       += self.data[i].image.nbytes
			pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)
			
	def load_image(self, index: int) -> tuple[np.ndarray, ImageInfo]:
		"""Load 1 image from dataset and preprocess image.

		Args:
			index (int):
				The image index.

		Returns:
			image (np.ndarray):
				The image.
			info (ImageInfo):
				The `ImageInfo` object.
		"""
		image = self.data[index].image
		info  = self.data[index].image_info
		
		if image is None:  # Not cached
			path  = self.image_paths[index]
			image = cv2.imread(path)  # BGR
			assert image is not None, f"Image not found at: {path}."
			
			# NOTE: Resize image while keeping the image ratio
			image, (h0, w0), (h1, w1) = resize_image(image, self.shape)
			
			# NOTE: Assign image info if it has not been defined
			#  (just to be sure)
			info        = ImageInfo.from_file(image_path=path, info=info)
			info.height = h1 if info.height != h1 else info.height
			info.width  = w1 if info.width  != w1 else info.width
			info.depth  = (image.shape[2] if info.depth != image.shape[2]
						   else info.depth)
			return image, info
		else:
			return self.data[index].image, self.data[index].image_info
	
	def cache_enhanced_images(self):
		"""Cache enhanced images into memory for faster training (WARNING:
		large datasets may exceed system RAM).
		"""
		n    = len(self.eimage_paths)
		gb   = 0  # Gigabytes of cached images
		pbar = tqdm(range(n), desc="Caching enhanced images")
		for i in pbar:  # Should be max 10k images
			# image, hw_original, hw_resized
			(self.data[i].eimage,
			 self.data[i].eimage_info) = self.load_enhanced_image(index=i)
			gb       += self.data[i].eimage.nbytes
			pbar.desc = "Caching enhanced images (%.1fGB)" % (gb / 1E9)
	
	def load_enhanced_image(self, index: int) -> tuple[np.ndarray, ImageInfo]:
		"""Load 1 enhanced image from dataset and preprocess image.

		Args:
			index (int):
				The enhanced image index.

		Returns:
			image (np.ndarray):
				The enhanced image.
			info (ImageInfo):
				The `ImageInfo` object.
		"""
		image = self.data[index].eimage
		info  = self.data[index].eimage_info
		
		if image is None:  # Not cached
			path  = self.eimage_paths[index]
			image = cv2.imread(path)  # BGR
			assert image is not None, f"Enhanced image not found at: {path}."
			
			# NOTE: Resize image while keeping the image ratio
			image, (h0, w0), (h1, w1) = resize_image(image, self.shape)
			
			# NOTE: Assign image info if it has not been defined
			# (just to be sure)
			info        = ImageInfo.from_file(image_path=path, info=info)
			info.height = h1 if info.height != h1 else info.height
			info.width  = w1 if info.width  != w1 else info.width
			info.depth  = (image.shape[2] if info.depth != image.shape[2]
						   else info.depth)
			return image, info
		else:
			return self.data[index].eimage, self.data[index].eimage_info
	
	# noinspection PyUnboundLocalVariable
	def load_mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
		"""Load 4 images and enhanced images and create a mosaic.
		
		Args:
			index (int):
				The image index.
				
		Returns:
			image4 (np.ndarray):
				The mosaic image.
			eimage4 (np.ndarray):
				The mosaic enhanced image.
		"""
		shape         = self.shape
		yc, xc        = shape[0], shape[1]  # Mosaic center x, y
		mosaic_border = [-yc // 2, -xc // 2]
		# 3 additional image indices
		indices = [index] + \
				  [random.randint(0, len(self.data) - 1) for _ in range(3)]
		
		# NOTE: Create mosaic image and enhanced image
		for i, index in enumerate(indices):
			# Load image
			image, info = self.load_image(index=index)
			eimage, _   = self.load_enhanced_image(index=index)
			h, w, _     = info.shape
			
			# Place image in image4
			if i == 0:  # Top left
				image4  = np.full((yc * 2, xc * 2, image.shape[2]),  114, np.uint8)  # base image with 4 tiles
				eimage4 = np.full((yc * 2, xc * 2, eimage.shape[2]), 114, np.uint8)  # base image with 4 tiles
				x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
				x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
			elif i == 1:  # Top right
				x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, xc * 2), yc
				x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
			elif i == 2:  # Bottom left
				x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc * 2, yc + h)
				x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
			elif i == 3:  # Bottom right
				x1a, y1a, x2a, y2a = xc, yc, min(xc + w, xc * 2), min(yc * 2, yc + h)
				x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

			image4[y1a:y2a, x1a:x2a]  = image[y1b:y2b, x1b:x2b]  # image4[ymin:ymax, xmin:xmax]
			eimage4[y1a:y2a, x1a:x2a] = eimage[y1b:y2b, x1b:x2b]    # image4[ymin:ymax, xmin:xmax]
			padw = x1a - x1b
			padh = y1a - y1b

		# NOTE: Augment
		image4, eimage4 = random_perspective_mask(
			image       = image4,
			mask        = eimage4,
			rotate      = self.augment.rotate,
			translate   = self.augment.translate,
			scale       = self.augment.scale,
			shear       = self.augment.shear,
			perspective = self.augment.perspective,
			border      = mosaic_border  # Border to remove
		)
		return image4, eimage4
		
	# MARK: Post-Load Data
	
	def post_load_data(self):
		"""Post load data operations. We prepare `batch_shapes` for
		`rect_training` augmentation, and some labels statistics. If you want
		to add more operations, just `extend` this method.
		"""
		# NOTE: Write data to our custom label format
		if not self.has_custom_labels or self.write_labels:
			self.write_custom_labels()

	# MARK: Utils
	
	def write_custom_labels(self):
		"""Write all data to custom label files using our custom label format.
		"""
		# NOTE: Get label files
		# label_files = [str(Path(eimage_file).stem) for eimage_file in self.eimage_files]
		# label_files = [f"{label_file}_custom.json" for label_file  in label_files]
		label_paths = self.label_paths
		dirnames    = [os.path.dirname(label_path)
					   for label_path in label_paths]
		create_dirs(paths=dirnames)
		
		# NOTE: Scan all images and enhanced images to get information
		for i in tqdm(range(len(self.data)), desc="Scanning images"):
			# image, hw_original, hw_resized
			_, self.data[i].image_info = self.load_image(index=i)
		
		for i in tqdm(range(len(self.data)), desc="Scanning enhanced images"):
			# image, hw_original, hw_resized
			_, self.data[i].eimage_info = self.load_enhanced_image(index=i)
		
		# NOTE: Parallel write labels
		pbar = tqdm(zip(self.data, label_paths),
					desc="Writing custom annotations", total=len(self.data))
		for (data, path) in pbar:
			VisualDataHandler().dump_to_file(data=data, path=path)
