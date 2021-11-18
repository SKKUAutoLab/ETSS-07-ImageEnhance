#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The formatter used to convert the data to our custom input format.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from torchkit.core.data import ObjectAnnotation as Annotation
from torchkit.core.image import augment_hsv
from torchkit.core.image import bbox_cxcywh_norm_xyxy
from torchkit.core.image import bbox_xyxy_cxcywh_norm
from torchkit.core.image import letterbox
from torchkit.core.image import random_perspective_bbox
from torchkit.core.image import random_perspective_mask
from torchkit.core.image import shift_bbox
from .base import BaseLabelFormatter
from .builder import LABEL_FORMATTERS


# MARK: - VisualDataFormatter

# noinspection PyAttributeOutsideInit
@LABEL_FORMATTERS.register(name="custom")
@LABEL_FORMATTERS.register(name="default")
@LABEL_FORMATTERS.register(name="visual")
class VisualDataFormatter(BaseLabelFormatter):
	"""The formatter for converting data to custom input."""

	# MARK: Get Item

	def get_classification_item(self, index: int) -> Any:
		"""Convert the data at the given index to classification input item."""
		raise RuntimeError("Visual data formatter do not provide "
						   "classification input.")

	def get_detection_item(
		self, index: int
	) -> tuple[np.ndarray, np.ndarray, tuple]:
		"""Convert the data at the given index to detection input item.

		Returns:
			image (np.ndarray):
				The image.
			labels_out (np.ndarray):
				The labels.
			shape (tuple):
				The shape of the resized image.
		"""
		# NOTE: Prepare data
		data               = self.data[index]
		self.load_image    = getattr(self.dataset, "load_image"   , None)
		self.load_mosaic   = getattr(self.dataset, "load_mosaic"  , None)
		self.image_size    = getattr(self.dataset, "image_size"   , None)
		self.batch_shapes  = getattr(self.dataset, "batch_shapes" , None)
		self.batch_indexes = getattr(self.dataset, "batch_indexes", None)
		assert getattr(data, "objects", None) is not None,\
			f"{data} doesn't have `objects` attribute."
		assert self.load_image is not None, \
			f"{self.dataset} does not have `load_image`."
		assert self.load_mosaic is not None, \
			f"{self.dataset} does not have `load_mosaic`."
		assert self.image_size is not None, \
			f"{self.dataset} does not have `image_size`."

		# NOTE: Load image mosaic
		if self.augment.mosaic and not self.augment.rect:
			image, labels = self.load_mosaic(index=index)
			shape         = image.shape
			# NOTE: MixUp https://arxiv.org/pdf/1710.09412.pdf
			if random.random() < self.augment.mixup:
				image2, labels2 = self.load_mosaic(
					index=random.randint(0, len(self.data) - 1)
				)
				ratio  = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
				image  = image * ratio + image2 * (1 - ratio)
				image  = image.astype(np.uint8)
				labels = np.concatenate((labels, labels2), 0)
				shape  = image.shape

		# NOTE: Load image normally
		else:
			image, info = self.load_image(index=index)
			(h0, w0, _) = info.shape0
			(h,  w,  _) = info.shape

			# Letterbox
			if self.augment.rect:
				shape = self.batch_shapes[self.batch_indexes[index]]
			else:
				shape = self.image_size

			scale_up = self.augment is not None
			image, ratio, pad = letterbox(image=image, new_shape=shape,
										  auto=False, scale_up=scale_up)
			shape = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

			# Load labels
			labels         = data.bbox_labels
			new_h          = ratio[1] * h
			new_w          = ratio[0] * w
			# Normalized xywh to xyxy format
			labels[:, 2:6] = bbox_cxcywh_norm_xyxy(labels[:, 2:6],
												   new_h, new_w)
			# Add padding
			labels[:, 2:6] = shift_bbox(labels[:, 2:6], pad[1], pad[0])

		# NOTE: Augmentation
		if self.augment is not None:
			# Augment imagespace
			if not self.augment.mosaic:
				image, labels = random_perspective_bbox(
					image       = image,
					bbox_labels = labels,
					rotate      = self.augment.rotate,
					translate   = self.augment.translate,
					scale       = self.augment.scale,
					shear       = self.augment.shear,
					perspective = self.augment.perspective,
				)
			# Augment colorspace
			augment_hsv(
				image = image,
				hgain = self.augment.hsv_h,
				sgain = self.augment.hsv_s,
				vgain = self.augment.hsv_v
			)
			# Apply cutouts
			# if random.random() < 0.9:
			#     bboxes = cutout_bbox(image, bboxes)
			# Normalize bounding boxes for flipping
			nl = len(labels)  # Number of labels
			if nl:
				labels = labels
			else:
				labels = np.zeros((nl, Annotation.bbox_label_len()))
			labels[:, 2:6] = bbox_xyxy_cxcywh_norm(labels[:, 2:6],
												   image.shape[0],
												   image.shape[1])
			# Flip up-down
			if random.random() < self.augment.flip_ud:
				image        = np.flipud(image)
				labels[:, 3] = 1 - labels[:, 3]
			# Flip left-right
			if random.random() < self.augment.flip_lr:
				image        = np.fliplr(image)
				labels[:, 2] = 1 - labels[:, 2]
			# Unnormalize bounding boxes
			# labels[:, 2:6] = bbox_cxcywh_norm_xyxy(labels[:, 2:6], image.shape[0], image.shape[1])

		# NOTE: Convert
		image  = image[:, :, ::-1]  # BGR to RGB
		image  = np.ascontiguousarray(image)
		labels = labels.astype(np.float32)
		return image, labels, shape

	def get_instance_item(self, index: int) -> Any:
		"""Convert the data at the given index to instance segmentation input
		item.
		"""
		raise RuntimeError("Visual data formatter do not provide instance "
						   "segmentation input.")

	def get_semantic_item(
		self, index: int
	) -> tuple[np.ndarray, np.ndarray, tuple]:
		"""Convert the data at the given index to semantic segmentation input
		item.

		Args:
			index (int):
				The item index.

		Returns:
			image (np.ndarray):
				The image.
			semantic (np.ndarray):
				The semantic segmentation mask.
			shape (tuple):
				The shape of the resized images.
		"""
		# NOTE: Prepare data
		self.load_image  = getattr(self.dataset, "load_image",  None)
		self.load_mosaic = getattr(self.dataset, "load_mosaic", None)
		self.load_semantic_image = getattr(
			self.dataset, "load_semantic_image", None
		)
		assert self.load_image is not None, \
			f"{self.dataset} does not have `load_image`."
		assert self.load_mosaic is not None,\
			f"{self.dataset} does not have `load_mosaic`."
		assert self.load_semantic_image is not None,\
			f"{self.dataset} does not have `load_semantic_image`."

		# NOTE: Load image mosaic
		if self.augment.mosaic and not self.augment.rect:
			image, semantic = self.load_mosaic(index=index)
			shape           = image.shape
			# NOTE: MixUp https://arxiv.org/pdf/1710.09412.pdf
			if random.random() < self.augment.mixup:
				image2, semantic2 = self.load_mosaic(
					index=random.randint(0, len(self.data) - 1)
				)
				# Mixup ratio, alpha=beta=8.0
				ratio    = np.random.beta(8.0, 8.0)
				image 	 = image * ratio + image2 * (1 - ratio)
				image    = image.astype(np.uint8)
				semantic = semantic * ratio + semantic2 * (1 - ratio)
				semantic = semantic.astype(np.uint8)
				shape 	 = image.shape

		# NOTE: Load image normally
		else:
			image, info = self.load_image(index=index)
			semantic, _ = self.load_semantic_image(index=index)
			(h0, w0, _) = info.shape0
			(h, w, _)   = info.shape
			shape       = (h0, w0), (h, w)

		# NOTE: Augmentation
		if self.augment is not None:
			# NOTE: Augment imagespace
			if not self.augment.mosaic:
				image, semantic = random_perspective_mask(
					image       = image,
					mask        = semantic,
					rotate      = self.augment.rotate,
					translate   = self.augment.translate,
					scale       = self.augment.scale,
					shear       = self.augment.shear,
					perspective = self.augment.perspective,
				)
			# NOTE: Augment colorspace
			augment_hsv(
				image = image,
				hgain = self.augment.hsv_h,
				sgain = self.augment.hsv_s,
				vgain = self.augment.hsv_v
			)
			# NOTE: Flip up-down
			if random.random() < self.augment.flip_ud:
				image    = np.flipud(image)
				semantic = np.flipud(semantic)
			# NOTE: Flip left-right
			if random.random() < self.augment.flip_lr:
				image    = np.fliplr(image)
				semantic = np.fliplr(semantic)

		# NOTE: Convert
		image    = image[:, :, ::-1]  # BGR to RGB
		image    = np.ascontiguousarray(image)
		semantic = semantic[:, :, ::-1]  # BGR to RGB
		semantic = np.ascontiguousarray(semantic)
		return image, semantic, shape

	def get_enhancement_item(
		self, index: int
	) -> tuple[np.ndarray, np.ndarray, tuple]:
		"""Convert the data at the given index to enhancement input item.

		Returns:
			image (np.ndarray):
				The image.
			eimage (np.ndarray):
				The enhanced image.
			shape (tuple):
				The shape of the resized images.
		"""
		# NOTE: Prepare data
		self.load_image  = getattr(self.dataset, "load_image",  None)
		self.load_mosaic = getattr(self.dataset, "load_mosaic", None)
		self.load_enhanced_image = getattr(self.dataset, "load_enhanced_image",
										   None)
		assert self.load_image is not None, \
			f"{self.dataset} does not have `load_image`."
		assert self.load_mosaic is not None, \
			f"{self.dataset} does not have `load_mosaic`."
		assert self.load_enhanced_image is not None, \
			f"{self.dataset} does not have `load_enhanced_image`."

		# NOTE: Load image mosaic
		if self.augment.mosaic and not self.augment.rect:
			image, eimage = self.load_mosaic(index=index)
			shape         = image.shape
			# NOTE: MixUp https://arxiv.org/pdf/1710.09412.pdf
			if random.random() < self.augment.mixup:
				image2, enh2 = self.load_mosaic(
					index=random.randint(0, len(self.data) - 1)
				)
				ratio  = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
				image  = image  * ratio + image2 * (1 - ratio)
				image  = image.astype(np.uint8)
				eimage = eimage * ratio + enh2 * (1 - ratio)
				eimage = eimage.astype(np.uint8)
				shape  = image.shape
		# NOTE: Load image normally
		else:
			image , info = self.load_image(index=index)
			eimage, _    = self.load_enhanced_image(index=index)
			(h0, w0, _)  = info.shape0
			(h,  w,  _)  = info.shape
			shape        = (h0, w0), (h, w)

		# NOTE: Augmentation
		if self.augment is not None:
			# NOTE: Augment imagespace
			if not self.augment.mosaic:
				image, eimage = random_perspective_mask(
					image       = image,
					mask        = eimage,
					rotate      = self.augment.rotate,
					translate   = self.augment.translate,
					scale       = self.augment.scale,
					shear       = self.augment.shear,
					perspective = self.augment.perspective,
				)
			# NOTE: Augment colorspace
			augment_hsv(
				image = image,
				hgain = self.augment.hsv_h,
				sgain = self.augment.hsv_s,
				vgain = self.augment.hsv_v
			)
			# NOTE: Flip up-down
			if random.random() < self.augment.flip_ud:
				image  = np.flipud(image)
				eimage = np.flipud(eimage)
			# NOTE: Flip left-right
			if random.random() < self.augment.flip_lr:
				image  = np.fliplr(image)
				eimage = np.fliplr(eimage)

		# NOTE: Convert
		image  = image[:, :, ::-1]  # BGR to RGB
		image  = np.ascontiguousarray(image)
		eimage = eimage[:, :, ::-1]  # BGR to RGB
		eimage = np.ascontiguousarray(eimage)
		return image, eimage, shape

	# MARK: Collate Function

	@staticmethod
	def collate_detection_fn(batch):
		image, labels, shapes = zip(*batch)  # transposed
		for i, l in enumerate(labels):
			l[:, 0] = i  # add target image index for build_targets()
		return torch.stack(image, 0), torch.cat(labels, 0), shapes

	@staticmethod
	def collate_semantic_fn(batch):
		image, semantic, shapes = zip(*batch)  # Transposed
		return torch.stack(image, 0), torch.stack(semantic, 0), shapes

	@staticmethod
	def collate_enhancement_fn(batch):
		image, eimage, shapes = zip(*batch)  # Transposed
		return torch.stack(image, 0), torch.stack(eimage, 0), shapes
