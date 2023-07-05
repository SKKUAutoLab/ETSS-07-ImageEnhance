#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for YOLO label/data format.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image

from torchkit.core.data import ImageInfo
from torchkit.core.data import ObjectAnnotation as Annotation
from torchkit.core.data import VisionData
from torchkit.core.fileio import dump
from torchkit.core.fileio import is_xml_file
from torchkit.core.fileio import load
from torchkit.core.image import bbox_area
from torchkit.core.image import bbox_cxcywh_norm_xyxy
from torchkit.core.image import bbox_xyxy_cxcywh_norm
from torchkit.core.image import exif_size
from .base import BaseLabelHandler
from .builder import LABEL_HANDLERS

logger = logging.getLogger()


# MARK: - PascalLabelHandler

@LABEL_HANDLERS.register(name="pascal")
class PascalLabelHandler(BaseLabelHandler):
	"""The handler for loading and dumping labels from Pascal label format to
	our custom label format defined in `torchkit.core.data.vision_data`.
	
	Pascal Format:
	
		<annotation>
			<folder>GeneratedData_Train</folder>
			<filename>000001.png</filename>
			<path>/my/path/GeneratedData_Train/000001.png</path>
			<source>
				<database>Unknown</database>
			</source>
			<size>
				<width>224</width>
				<height>224</height>
				<depth>3</depth>
			</size>
			<segmented>0</segmented>
			<object>
				<name>21</name>
				<pose>Frontal</pose>
				<truncated>0</truncated>
				<difficult>0</difficult>
				<occluded>0</occluded>
				<bndbox>
					<xmin>82</xmin>
					<xmax>172</xmax>
					<ymin>88</ymin>
					<ymax>146</ymax>
				</bndbox>
			</object>
		</annotation>
	
	- name:
		This is the name of the object that we are trying to identify
		(i.e., class_id).
	- truncated:
		Indicates that the bounding box specified for the object does not
		correspond to the full extent of the object. For example, if an
		object is visible partially in the image then we set truncated to 1.
		If the object is fully visible then set truncated to 0.
	- difficult:
		An object is marked as difficult when the object is considered
		difficult to recognize. If the object is difficult to recognize then
		we set difficult to 1 else set it to 0.
	- bndbox:
		Axis-aligned rectangle specifying the extent of the object visible in
		the image.
	"""
	
	# MARK: Load
	
	def load_from_file(
		self, image_path: str, label_path: str, **kwargs
	) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				The image filepath.
			label_path (str):
				The label filepath.
				
		Return:
			visual_data (VisionData):
				A `VisualData` item.
		"""
		# NOTE: Get image shape
		image = Image.open(image_path)
		image.verify()  # PIL verify
		shape0 = exif_size(image)  # Image size (height, width)
		assert (shape0[0] > 9) & (shape0[1] > 9), \
			f"{image_path}: image size <10 pixels."
		
		# NOTE: Load content from file
		label_dict = load(label_path) if is_xml_file(label_path) else None
		label_dict = label_dict.get("annotation")
		folder     = label_dict.get("folder")
		filename   = label_dict.get("filename")
		path       = label_dict.get("path")
		size       = label_dict.get("size")
		height     = size.get("height")
		width      = size.get("width")
		depth      = size.get("depth")
		segmented  = label_dict.get("segmented")
		obj        = label_dict.get("object")
		
		# NOTE: Parse image info
		common_prefix = os.path.commonprefix([image_path, label_path])
		stem          = str(Path(image_path).stem)
		
		info         = ImageInfo()
		info.id      = info.id   if (info.id != stem) else stem
		info.name    = (filename if (filename is not None)
						else str(Path(image_path).name))
		info.path    = image_path.replace(common_prefix, "")
		info.height0 = height if (height is not None) else shape0[0]
		info.width0  = width  if (width is not None)  else shape0[1]
		info.depth   = depth  if (depth is not None)  else info.depth
		
		# NOTE: Parse all annotations
		objs = []
		for i, l in enumerate(obj):
			height0   = info.height0
			width0    = info.width0
			class_id  = l.get("name")
			pose      = l.get("pose")
			truncated = l.get("truncated")
			difficult = l.get("difficult")
			occluded  = l.get("occluded")
			bndbox    = l.get("bndbox")
			bbox_xyxy = np.array([bndbox["xmin"], bndbox["ymin"],
								  bndbox["xmax"], bndbox["ymax"]], np.float32)
			bbox_cxcywh_norm = bbox_xyxy_cxcywh_norm(bbox_xyxy, height0, width0)
			area 			 = bbox_area(bbox_xyxy)
			objs.append(
				Annotation(
					class_id   = int(class_id),
					bbox       = bbox_cxcywh_norm,
					area       = area,
					truncation = truncated,
					occlusion  = occluded,
					difficult  = difficult
				)
			)
			
		return VisionData(image_info=info, objects=objs)
	
	# MARK: Dump
	
	def dump_to_file(self, data: VisionData, path: str, **kwargs):
		"""Dump data from object to file.
		
		Args:
			data (VisionData):
				The `VisualData` item.
			path (str):
				The label filepath to dump the data.
		"""
		# NOTE: Prepare output data
		label_dict              = OrderedDict()
		info                    = data.image_info
		label_dict["folder"]    = str(Path(info.path).parent)
		label_dict["filename"]  = info.name
		label_dict["path"]      = info.path
		label_dict["source"]    = {"database": "Unknown"}
		label_dict["size"] 	    = {"width" : info.width0,
								   "height": info.height0,
								   "depth" : info.depth}
		label_dict["segmented"] = 0
		
		objs = []
		for obj in data.objects:
			xyxy = bbox_cxcywh_norm_xyxy(obj.bbox, info.height0, info.width0)
			ann_dict = {
				"name"     : obj.class_id,
				"pose"     : "Unknown",
				"truncated": int(obj.truncation),
				"difficult": int(obj.difficult),
				"occluded" : int(obj.occlusion),
				"bndbox"   : {"xmin": int(xyxy[0]),
							  "xmax": int(xyxy[1]),
							  "ymin": int(xyxy[2]),
							  "ymax": int(xyxy[3])}
			}
			objs.append(ann_dict)
			
		label_dict["object"]     = objs
		label_dict["annotation"] = label_dict
		
		# NOTE: Dump to file
		dump(obj=label_dict, path=path, file_format="xml")
