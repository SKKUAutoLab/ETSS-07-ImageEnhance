#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for Cityscapes label/data format.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
from torchkit.core.data import ClassLabels
from torchkit.core.data import ImageInfo
from torchkit.core.data import ObjectAnnotation as Annotation
from torchkit.core.data import VisionData
from torchkit.core.fileio import dump
from torchkit.core.fileio import is_json_file
from torchkit.core.fileio import load

from torchkit.core.image import bbox_area
from torchkit.core.image import bbox_xywh_xyxy
from torchkit.core.image import bbox_xyxy_cxcywh_norm
from .base import BaseLabelHandler
from .builder import LABEL_HANDLERS

logger = logging.getLogger()


# MARK: - CityscapesLabelHandler

@LABEL_HANDLERS.register(name="cityscapes")
class CityscapesLabelHandler(BaseLabelHandler):
	"""The handler for loading and dumping labels from Cityscapes label format
	to our custom label format defined in `torchkit.core.data.vision_data`.
	
	Cityscapes format:
		{
			"imgHeight": ...
			"imgWidth": ...
			"objects": [
				{
					"labels": ...
					"polygon": [
						[<x>, <y>],
						...
					]
				}
				...
			]
		}
	"""
	
	# MARK: Load
	
	def load_from_file(
		self,
		image_path : str,
		label_path : str,
		classlabels: Optional[ClassLabels] = None,
		**kwargs
	) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				The image filepath.
			label_path (str):
				The label filepath.
			classlabels (ClassLabels, optional):
				The `ClassLabels` object contains all class-labels defined in
				the dataset.
				
		Return:
			visual_data (VisionData):
				A `VisualData` item.
		"""
		# NOTE: Load content from file
		label_dict = load(label_path) if is_json_file(label_path) else None
		height0    = label_dict.get("imgHeight")
		width0     = label_dict.get("imgWidth")
		objects    = label_dict.get("objects")
		
		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path)

		if height0 != image_info.height0:
			image_info.height0 = height0
		else:
			image_info.height0 = image_info.height0

		if width0 != image_info.width0:
			image_info.width0 = width0
		else:
			image_info.width0 = image_info.width0

		# NOTE: Parse all annotations
		objs = []
		for i, l in enumerate(objects):
			label = l.get("label")
			# If the label is not known, but ends with a 'group'
			# (e.g. cargroup) try to remove the s and see if that works
			if (classlabels.get_id(name=label) is None) and	\
				label.endswith("group"):
				label = label[:-len("group")]
			if classlabels is None:
				class_id = label
			else:
				class_id = classlabels.get_id(name=label)
			polygon          = l.get("polygon", None)
			bbox_xywh        = cv2.boundingRect(polygon)
			bbox_xywh        = np.array(bbox_xywh, np.float32)
			bbox_xyxy        = bbox_xywh_xyxy(bbox_xywh)
			bbox_cxcywh_norm = bbox_xyxy_cxcywh_norm(bbox_xyxy, height0, width0)
			area             = bbox_area(bbox_xyxy)
			annotation 		 = Annotation(
				class_id = class_id,
				bbox     = bbox_cxcywh_norm,
				area     = area,
			)
			if polygon is not None:
				annotation.polygon_vertices = [polygon]
			objs.append(annotation)
			
		return VisionData(image_info=image_info, objects=objs)
	
	# MARK: Dump
	
	def dump_to_file(
		self,
		data       : VisionData,
		path       : str,
		classlabels: Optional[ClassLabels] = None,
		**kwargs
	):
		"""Dump data from object to file.
		
		Args:
			data (VisionData):
				The `VisualData` item.
			path (str):
				The label filepath to dump the data.
			classlabels (ClassLabels, optional):
				The `ClassLabels` object contains all class-labels defined in
				the dataset. Default: `None`.
		"""
		# NOTE: Prepare output data
		label_dict              = OrderedDict()
		info                    = data.image_info
		label_dict["imgHeight"] = info.height0
		label_dict["imgWidth"]  = info.width0
		
		objs = []
		for obj in data.objects:
			if classlabels is None:
				label = obj.class_id
			else:
				label = classlabels.get_name(key="id", value=obj.class_id)
			obj_dict = {
				"label"  : label,
				"polygon": obj.polygon_vertices[0]
			}
			objs.append(obj_dict)
		label_dict["objects"] = objs
		
		# NOTE: Dump to file
		dump(obj=label_dict, path=path, file_format="json")
