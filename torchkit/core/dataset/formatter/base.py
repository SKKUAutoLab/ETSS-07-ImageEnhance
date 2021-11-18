#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for different formatters. Each subclass formatter is used to
convert the data to different input formats (for training). For example:
yolo-based model input, classification input, etc.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

from torchkit.core.data import ImageAugment

logger = logging.getLogger()


# MARK: - BaseLabelFormatter

class BaseLabelFormatter(metaclass=ABCMeta):
	"""The template for converting data to different input formats.
	
	Attributes:
		dataset (object):
			The dataset object.
		data (list, optional):
			The list of data items.
		augment (object, optional):
			A object contains all hyperparameters for augmentation operations.
	"""
	
	# MARK: Magic Function
	
	def __init__(self, dataset: object):
		self.dataset = dataset

		self.data = None
		if hasattr(self.dataset, "data"):
			self.data = self.dataset.data

		self.augment = None
		if hasattr(self.dataset, "augment"):
			self.augment = self.dataset.augment

		assert isinstance(self.data, list) and len(self.data) >= 0, \
			f"No data available."
		assert isinstance(self.augment, ImageAugment), \
			f"{self.augment} is not a `VisualAugment` object."
	
	# MARK: Get Item
	
	@abstractmethod
	def get_classification_item(self, index: int) -> Any:
		"""Convert the data at the given index to classification input item."""
		pass
	
	@abstractmethod
	def get_detection_item(self, index: int) -> Any:
		"""Convert the data at the given index to detection input item."""
		pass
	
	@abstractmethod
	def get_instance_item(self, index: int) -> Any:
		"""Convert the data at the given index to instance segmentation input
		item.
		"""
		pass

	@abstractmethod
	def get_semantic_item(self, index: int) -> Any:
		"""Convert the data at the given index to semantic segmentation input
		item.
		"""
		pass
	
	@abstractmethod
	def get_enhancement_item(self, index: int) -> Any:
		"""Convert the data at the given index to enhancement input item."""
		pass
