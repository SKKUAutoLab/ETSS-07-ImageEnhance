#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class for several vision tasks such as image classification, object
detection, segmentation, ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Union

import numpy as np

from torchkit.core.utils import ID
from .image_info import ImageInfo
from .object_annotation import ObjectAnnotation

logger = logging.getLogger()


# MARK: - VisionData

def make_default_objects() -> list[ObjectAnnotation]:
	"""Returns an empty list of object annotations."""
	return []


@dataclass
class VisionData:
	"""Visual Data implements a data class for all vision tasks. This is a
	generalization of the COCO format.
	
	Attributes:
		image (np.ndarray, optional):
			The image.
		image_info (torchkit.core.data.image_info.ImageInfo):
			Image information.
		image_annotation (int, optional):
			The image annotation/label (usually, `class_id`). This is used
			for classification task.
		semantic (np.ndarray, optional):
			Each pixel has an ID that represents the ground truth label. This
			is used for semantic segmentation task.
		semantic_info (ImageInfo, optional):
			The semantic segmentation mask information.
		instance (np.ndarray, optional):
			The pixel values encode both, class and the individual instance.
			Let's say your labels.py assigns the ID 26 to the class `car`.
			Then, the individual cars in an image get the IDs 26000, 26001,
			26002, ... . A group of cars, where our annotators could not
			identify the individual instances anymore, is assigned to the ID 26.
			This is used for instance segmentation class.
		instance_info (ImageInfo, optional):
			The instance image information.
		panoptic (np.ndarray, optional):
			This is used for panoptic segmentation task.
		panoptic_info (ImageInfo, optional):
			The panoptic image information.
		eimage (np.ndarray, optional):
			Short for "enhanced image". The good quality image. When this is
			used, `image` will have poor quality. This is used in image
			enhancement task.
		eimage_info (ImageInfo, optional):
			The enhanced image information.
		objects (list):
			The list of all object annotations in the image. This is used for
			object detection and instance segmentation tasks.
	
	References:
		https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
	"""

	image           : Optional[np.ndarray]   = None
	image_info      : ImageInfo				 = ImageInfo()
	image_annotation: Union[ID, None]  		 = None
	semantic        : Optional[np.ndarray]   = None
	semantic_info   : Optional[ImageInfo]    = None
	instance        : Optional[np.ndarray]   = None
	instance_info   : Optional[ImageInfo]    = None
	panoptic        : Optional[np.ndarray]   = None
	panoptic_info   : Optional[ImageInfo]    = None
	eimage          : Optional[np.ndarray]   = None
	eimage_info     : Optional[ImageInfo]    = None
	objects         : list[ObjectAnnotation] = field(
		default_factory=make_default_objects
	)
	
	# MARK: Properties
	
	@property
	def bbox_labels(self) -> np.ndarray:
		"""Return bounding box labels for detection task:
		<image_id> <class_id> <x1> <y1> <x2> <y2> <confidence> <area>
		<truncation> <occlusion>
		"""
		return np.array(
			[obj.bbox_label for obj in self.objects], dtype=np.float32
		)
	
	@property
	def class_id(self) -> Optional[ID]:
		"""An alias of `image_annotation`."""
		return self.image_annotation
