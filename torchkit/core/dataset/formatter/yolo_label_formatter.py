#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The formatter is used to convert the data to YOLO detection input.
"""

from __future__ import annotations

import logging
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
from torchkit.core.image import shift_bbox
from .base import BaseLabelFormatter
from .builder import LABEL_FORMATTERS

logger = logging.getLogger()


# MARK: - YoloLabelFormatter

# noinspection PyAttributeOutsideInit
@LABEL_FORMATTERS.register(name="yolo")
class YoloLabelFormatter(BaseLabelFormatter):
	"""The formatter for converting data to YOLO detection input."""

	# MARK: Get Item
	
	def get_classification_item(self, index: int) -> Any:
		"""Convert the data at the given index to classification input item."""
		raise RuntimeError("Yolo label formatter do not provide "
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
		assert getattr(data, "objects", None) is not None, \
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
			# MixUp https://arxiv.org/pdf/1710.09412.pdf
			if random.random() < self.augment.mixup:
				image2, labels2 = self.load_mosaic(
					index=random.randint(0, len(self.data) - 1)
				)
				# Mixup ratio, alpha=beta=8.0
				ratio  = np.random.beta(8.0, 8.0)
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
			scale_up		  = self.augment is not None
			image, ratio, pad = letterbox(image=image, new_shape=shape,
										  auto=False, scale_up=scale_up)
			# For COCO mAP rescaling
			shape = (h0, w0), ((h / h0, w / w0), pad)

			# Load labels
			labels         = data.bbox_labels
			new_h          = ratio[1] * h
			new_w          = ratio[0] * w
			labels[:, 2:6] = bbox_cxcywh_norm_xyxy(labels[:, 2:6],
												   new_h, new_w)
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
			
		# NOTE: Convert
		image  = image[:, :, ::-1]  # BGR to RGB
		image  = np.ascontiguousarray(image)
		labels = labels.astype(np.float32)
		return image, labels, shape
	
	def get_instance_item(self, index: int) -> Any:
		""""Convert the data at the given index to instance segmentation input
		item.
		"""
		raise RuntimeError("Yolo label formatter do not provide instance "
						   "segmentation input.")
	
	def get_semantic_item(self, index: int) -> Any:
		"""Convert the data at the given index to semantic segmentation input
		item.
		"""
		raise RuntimeError("Yolo label formatter do not provide semantic "
						   "segmentation input.")
	
	def get_enhancement_item(self, index: int) -> Any:
		"""Convert the data at the given index to enhancement input item."""
		raise RuntimeError("Yolo label formatter do not provide enhancement "
						   "input.")

	# MARK: Collate Function
	
	@staticmethod
	def collate_detection_fn(batch):
		image, labels, shapes = zip(*batch)  # transposed
		for i, l in enumerate(labels):
			l[:, 0] = i  # add target image index for build_targets()
		return torch.stack(image, 0), torch.cat(labels, 0), shapes
