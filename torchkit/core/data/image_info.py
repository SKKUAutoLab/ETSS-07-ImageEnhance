#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class for storing image info.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from pathlib import Path
from typing import Optional
from typing import Optional
from typing import Union

import numpy as np
from PIL import Image
from PIL import Image

from torchkit.core.utils import Dim3
from torchkit.core.utils import Dim3
from torchkit.core.utils import ID
from torchkit.core.utils import ID

logger = logging.getLogger()


# MARK: - ImageInfo

@dataclass
class ImageInfo:
	"""ImageInfo is a data class for storing image information.
	
	Attributes:
		id (ID):
			The image ID. This attribute is useful for batch processing but you
			want to keep the objects in the correct frame sequence.
		name (str):
			The image name with extension.
		path (str):
			The image path.
		height0 (int):
			The original image height.
		width0 (int):
			The original image width.
		height (int):
			The resized image height.
		width (int):
			The resized image width.
		depth (int):
			The image channels.
		
	References:
		https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
	"""

	id     : ID  = uuid.uuid4().int
	name   : str = ""
	path   : str = ""
	height0: int = 0
	width0 : int = 0
	height : int = 0
	width  : int = 0
	depth  : int = 3

	# MARK: Configure

	@staticmethod
	def from_file(
		image_path: str, info: Optional[ImageInfo] = None
	) -> ImageInfo:
		"""Parse image info from image file.

		Args:
			image_path (str):
				The image path.
			info (ImageInfo, optional):
				The `ImageInfo` object.
				
		Returns:
			info (ImageInfo):
				The `ImageInfo` object.
		"""
		from torchkit.core.image import exif_size
		
		# NOTE: Get image shape
		image = Image.open(image_path)
		image.verify()  # PIL verify
		shape0 = exif_size(image)  # Image size (height, width)
		assert (shape0[0] > 9) & (shape0[1] > 9), \
			f"{image_path}: image size <10 pixels."
		
		# NOTE: Parse image info
		path = Path(image_path)
		stem = str(path.stem)
		name = str(path.name)
		
		info         = ImageInfo() if info is None               else info
		info.id      = stem        if info.id      != stem       else info.id
		info.name    = name        if info.name    != name       else info.name
		info.path    = image_path  if info.path    != image_path else info.path
		info.height0 = shape0[0]   if info.height0 != shape0[0]  else info.height0
		info.width0  = shape0[1]   if info.width0  != shape0[1]  else info.width0
		info.height  = shape0[0]   if info.height  != shape0[0]  else info.height
		info.width   = shape0[1]   if info.width   != shape0[1]  else info.width
		return info
	
	# MARK: Properties
	
	@property
	def shape0(self) -> Dim3:
		"""Return the image's original shape [H, W, C]."""
		return self.height0, self.width0, self.depth
	
	@shape0.setter
	def shape0(self, value: tuple):
		"""Assign the image's original shape."""
		if len(value) == 3:
			self.height0, self.width0, self.depth = value[0], value[1], value[2]
		elif len(value) == 2:
			self.height0, self.width0 = value[0], value[1]
	
	@property
	def shape(self) -> tuple:
		"""Return the image's resized shape [H, W, C]."""
		return self.height, self.width, self.depth
	
	@shape.setter
	def shape(self, value: tuple):
		"""Assign the image's resized shape [H, W, C]."""
		if len(value) == 3:
			self.height, self.width, self.depth = value[0], value[1], value[2]
		elif len(value) == 2:
			self.height, self.width = value[0], value[1]
