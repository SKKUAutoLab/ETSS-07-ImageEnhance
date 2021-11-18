#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data classes for storing of augmentation configs for vision tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from torchkit.core.fileio import load

logger = logging.getLogger()


# MARK: - ImageAugment

@dataclass
class ImageAugment:
	"""Data classes for storing of augmentation configs for vision tasks.
	
	Attributes:
		hsv_h (float):
			Image HSV-Hue augmentation (fraction).
		hsv_s (float):
			Image HSV-Saturation augmentation (fraction).
		hsv_v (float):
			Image HSV-Value augmentation (fraction).
		rotate (float):
			Image rotation (+/- deg).
		translate (float):
			Image translation (+/- fraction).
		scale (float):
			Image scale (+/- gain).
		shear (float):
			Image shear (+/- deg).
		perspective (float):
			Image perspective (+/- fraction), range 0-0.001.
		flip_ud (float):
			Image flip up-down (probability).
		flip_lr (float):
			Image flip left-right (probability).
		mixup (float):
			Image mixup (probability).
		mosaic (float):
			Use mosaic augmentation.
		rect_training (float):
			Train model using rectangular images instead of square ones.
		stride (float):
			When `rect_training=True`, reshape the image shapes as a multiply
			of stride.
		pad (float):
			When `rect_training=True`, pad the empty pixel with given values.
	"""
	
	hsv_h      : float = 0.0
	hsv_s      : float = 0.0
	hsv_v      : float = 0.0
	rotate     : float = 0.0
	translate  : float = 0.0
	scale      : float = 0.0
	shear      : float = 0.0
	perspective: float = 0.0
	flip_ud    : float = 0.0
	flip_lr    : float = 0.0
	mixup      : float = 0.0
	mosaic     : bool  = False
	rect       : bool  = False
	stride     : int   = 32
	pad        : float = 0.0

	# MARK: Configure

	@classmethod
	def from_file(cls, path: str):
		"""Create a `ImageAugment` object from file.
		
		Args:
			path (str):
				The filepath to the config file that stores the augmentation
				configs.
		"""
		aug = load(path=path)
		assert aug is not None, f"No configs can be found in {path}."
		return cls(**aug)
