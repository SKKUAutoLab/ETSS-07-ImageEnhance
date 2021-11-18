#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for bounding box. For example: format conversion, geometric
calculations, box metrics, ...
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torchkit.core.utils import Dim2

from .imageproc import to_channel_last
from .imageproc import unnormalize_image

logger = logging.getLogger()


# MARK: - Operations on Box

def bbox_area(xyxy: np.ndarray) -> float:
	"""Calculate the bbox area."""
	return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
	

def bbox_candidates(
	xyxy1   : np.ndarray,
	xyxy2   : np.ndarray,
	wh_thr  : float = 2,
	ar_thr  : float = 20,
	area_thr: float = 0.2
) -> bool:
	"""Return `True` if xyxy2 is the candidate for xyxy1
	
	Args:
		xyxy1 (np.ndarray):
			The bounding box before augment as [x1, y1, x2, y2].
		xyxy2 (np.ndarray):
			The bounding box after augment as [x1, y1, x2, y2].
		wh_thr (float):
			Threshold of both width and height (pixels).
		ar_thr (float):
			Aspect ratio threshold.
		area_thr (float):
			Area ratio threshold.
	"""
	w1, h1 = xyxy1[2] - xyxy1[0], xyxy1[3] - xyxy1[1]
	w2, h2 = xyxy2[2] - xyxy2[0], xyxy2[3] - xyxy2[1]
	ar     = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # Aspect ratio
	return ((w2 > wh_thr) &
			(h2 > wh_thr) &
			(w2 * h2 / (w1 * h1 + 1e-16) > area_thr) &
			(ar < ar_thr))  # candidates


def bbox_ioa(xyxy1: np.ndarray, xyxy2: np.ndarray) -> np.ndarray:
	"""Calculate the intersection over area given xyxy1, xyxy2.
	
	Args:
		xyxy1 (np.ndarray):
			A single bounding box as [x1, y1, x2, y2].
		xyxy2 (np.ndarray):
			An array of bounding boxes as [:, x1, y1, x2, y2].
			
	Returns:
		ioa (np.ndarray):
			The ioa metrics.
	"""
	xyxy2 = xyxy2.transpose()
	
	# Get the coordinates of bounding boxes
	b1_x1, b1_y1, b1_x2, b1_y2 = xyxy1[0], xyxy1[1], xyxy1[2], xyxy1[3]
	b2_x1, b2_y1, b2_x2, b2_y2 = xyxy2[0], xyxy2[1], xyxy2[2], xyxy2[3]
	
	# Intersection area
	inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
	             (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
		
	# bbox2 area
	bbox2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
	
	# Intersection over box2 area
	return inter_area / bbox2_area


@dispatch(np.ndarray, np.ndarray)
def bbox_iou(xyxy1: np.ndarray, xyxy2: np.ndarray) -> float:
	"""Find the Intersection over Union (IoU) between two 2 boxes.

	Args:
		xyxy1 (np.ndarray):
			The target bounding box as [x1, y1, x2, y2].
		xyxy2 (np.ndarray):
			The ground-truth bounding box as [x1, y1, x2, y2].

	Returns:
		iou (float):
			The ratio IoU.
	"""
	xx1 = np.maximum(xyxy1[0], xyxy2[0])
	yy1 = np.maximum(xyxy1[1], xyxy2[1])
	xx2 = np.minimum(xyxy1[2], xyxy2[2])
	yy2 = np.minimum(xyxy1[3], xyxy2[3])
	w   = np.maximum(0.0, xx2 - xx1)
	h   = np.maximum(0.0, yy2 - yy1)
	wh  = w * h
	ou  = wh / ((xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1]) +
				(xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1]) - wh)
	return ou


@dispatch(torch.Tensor, torch.Tensor)
def bbox_iou(xyxy1: torch.Tensor, xyxy2: torch.Tensor) -> float:
	"""Find the Intersection over Union (IoU) between two 2 boxes.

	Args:
		xyxy1 (torch.Tensor):
			The target bounding box as [x1, y1, x2, y2].
		xyxy2 (torch.Tensor):
			The ground-truth bounding box as [x1, y1, x2, y2].

	Returns:
		iou (float):
			The ratio IoU.
	"""
	xyxy1_np = xyxy1.numpy()
	xyxy2_np = xyxy2.numpy()
	return bbox_iou(xyxy1_np, xyxy2_np)


@dispatch(np.ndarray, np.ndarray)
def batch_bbox_iou(xyxy1: np.ndarray, xyxy2: np.ndarray) -> np.ndarray:
	"""From SORT: Computes IOU between two sets of boxes.

	Args:
		xyxy1 (np.ndarray):
			The target bounding boxes as [x1, y1, x2, y2].
		xyxy2 (np.ndarray):
			The ground-truth bounding boxes as [x1, y1, x2, y2].

	Returns:
		iou (np.ndarray):
			The ratio IoUs.
	"""
	xyxy1 = np.expand_dims(xyxy1, 1)
	xyxy2 = np.expand_dims(xyxy2, 0)
	xx1   = np.maximum(xyxy1[..., 0], xyxy2[..., 0])
	yy1   = np.maximum(xyxy1[..., 1], xyxy2[..., 1])
	xx2   = np.minimum(xyxy1[..., 2], xyxy2[..., 2])
	yy2   = np.minimum(xyxy1[..., 3], xyxy2[..., 3])
	w     = np.maximum(0.0, xx2 - xx1)
	h     = np.maximum(0.0, yy2 - yy1)
	wh    = w * h
	iou   = wh / ((xyxy1[..., 2] - xyxy1[..., 0]) *
				  (xyxy1[..., 3] - xyxy1[..., 1]) +
				  (xyxy2[..., 2] - xyxy2[..., 0]) *
				  (xyxy2[..., 3] - xyxy2[..., 1]) - wh)
	return iou


# MARK: - Box Transformation

def clip_bbox_xyxy(xyxy: torch.Tensor, image_size: Dim2) -> torch.Tensor:
	"""Clip bounding boxes to image size [H, W].

	Args:
		xyxy (torch.Tensor):
			The bounding boxes coordinates as [x1, y1, x2, y2].
		image_size (tuple):
			The image size as [H, W].

	Returns:
		box_xyxy (torch.Tensor):
			The clipped bounding boxes.
	"""
	xyxy[:, 0].clamp_(0, image_size[1])  # x1
	xyxy[:, 1].clamp_(0, image_size[0])  # y1
	xyxy[:, 2].clamp_(0, image_size[1])  # x2
	xyxy[:, 3].clamp_(0, image_size[0])  # y2
	return xyxy


@dispatch(np.ndarray, np.ndarray)
def cutout_bbox(
	image: np.ndarray, bbox_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
	"""Applies image cutout augmentation with bounding box labels.

	References:
		https://arxiv.org/abs/1708.04552

	Args:
		image (np.ndarray):
			The image.
		bbox_labels (np.ndarray):
			The bounding box labels where the bbox coordinates are located at:
			labels[:, 2:6].

	Returns:
		image_cutout (np.ndarray):
			The cutout image.
		bbox_labels_cutout (np.ndarray):
			The cutout labels.
	"""
	h, w               = image.shape[:2]
	image_cutout       = image.copy()
	bbox_labels_cutout = bbox_labels.copy()
	
	# NOTE: Create random masks
	scales = [0.5] * 1 + \
			 [0.25] * 2 + \
			 [0.125] * 4 + \
			 [0.0625] * 8 + \
			 [0.03125] * 16  # image size fraction
	for s in scales:
		mask_h = random.randint(1, int(h * s))
		mask_w = random.randint(1, int(w * s))
		
		# Box
		xmin = max(0, random.randint(0, w) - mask_w // 2)
		ymin = max(0, random.randint(0, h) - mask_h // 2)
		xmax = min(w, xmin + mask_w)
		ymax = min(h, ymin + mask_h)
		
		# Apply random color mask
		image_cutout[ymin:ymax, xmin:xmax] = [random.randint(64, 191)
											  for _ in range(3)]
		
		# Return unobscured bounding boxes
		if len(bbox_labels_cutout) and s > 0.03:
			box = np.array([xmin, ymin, xmax, ymax], np.float32)
			ioa = bbox_ioa(box, bbox_labels_cutout[:, 2:6])  # Intersection over area
			bbox_labels_cutout = bbox_labels_cutout[ioa < 0.60]  # Remove >60% obscured labels
	
	return image_cutout, bbox_labels_cutout


def scale_bbox_xyxy(
	xyxy         : torch.Tensor,
	detector_size: Dim2,
	original_size: Dim2,
	ratio_pad     = None
) -> torch.Tensor:
	"""Scale bounding boxes coordinates (from detector size) to the original
	image size.

	Args:
		xyxy (torch.Tensor):
			The bounding boxes coordinates as [x1, y1, x2, y2].
		detector_size (tuple):
			The detector's input size as [H, W].
		original_size (tuple):
			The original image size as [H, W].
		ratio_pad:

	Returns:
		box_xyxy (torch.Tensor):
			The scaled bounding boxes.
	"""
	if ratio_pad is None:  # Calculate from original_size
		gain = min(detector_size[0] / original_size[0],
				   detector_size[1] / original_size[1])  # gain  = old / new
		pad  = (detector_size[1] - original_size[1] * gain) / 2, \
			   (detector_size[0] - original_size[0] * gain) / 2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad  = ratio_pad[1]
	
	xyxy[:, [0, 2]] -= pad[0]  # x padding
	xyxy[:, [1, 3]] -= pad[1]  # y padding
	xyxy[:, :4]     /= gain
	return clip_bbox_xyxy(xyxy, original_size)


def random_perspective_bbox(
	image      : np.ndarray,
	bbox_labels: np.ndarray         = (),
	rotate     : float              = 10,
	translate  : float              = 0.1,
	scale      : float              = 0.1,
	shear      : float              = 10,
	perspective: float              = 0.0,
	border     : Union[tuple, list] = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
	"""Perform random perspective the image and the corresponding bounding box
	labels.

	Args:
		image (np.ndarray):
			The image.
		bbox_labels (np.ndarray):
			The bounding box labels where the bbox coordinates are located at:
			labels[:, 2:6]. Default: `()`.
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
		border (tuple, list):

	Returns:
		image_new (np.ndarray):
			The augmented image.
		bbox_labels_new (np.ndarray):
			The augmented bounding boxes.
	"""
	# torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
	# targets = [cls, xyxy]
	
	height          = image.shape[0] + border[0] * 2  # Shape of [HWC]
	width           = image.shape[1] + border[1] * 2
	image_new       = image.copy()
	bbox_labels_new = bbox_labels.copy()
	
	# NOTE: Center
	C       = np.eye(3)
	C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
	C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
	
	# NOTE: Perspective
	P       = np.eye(3)
	P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
	P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
	
	# NOTE: Rotation and Scale
	R = np.eye(3)
	a = random.uniform(-rotate, rotate)
	# a += random.choice([-180, -90, 0, 90])  # Add 90deg rotations to small rotations
	s = random.uniform(1 - scale, 1 + scale)
	# s = 2 ** random.uniform(-scale, scale)
	R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
	
	# NOTE: Shear
	S       = np.eye(3)
	S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
	S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
	
	# NOTE: Translation
	T       = np.eye(3)
	T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
	T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
	
	# NOTE: Combined rotation matrix
	M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
	if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # Image changed
		if perspective:
			image_new = cv2.warpPerspective(image_new, M, dsize=(width, height),
											borderValue=(114, 114, 114))
		else:  # Affine
			image_new = cv2.warpAffine(image_new, M[:2], dsize=(width, height),
									   borderValue=(114, 114, 114))
	
	# Visualize
	# import matplotlib.pyplot as plt
	# ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
	# ax[0].imshow(img[:, :, ::-1])  # base
	# ax[1].imshow(img2[:, :, ::-1])  # warped
	
	# NOTE: Transform bboxes' coordinates
	n = len(bbox_labels_new)
	if n:
		# NOTE: Warp points
		xy = np.ones((n * 4, 3))
		xy[:, :2] = bbox_labels_new[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
		xy = xy @ M.T  # Transform
		if perspective:
			xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # Rescale
		else:  # Affine
			xy = xy[:, :2].reshape(n, 8)
		
		# NOTE: Create new boxes
		x  = xy[:, [0, 2, 4, 6]]
		y  = xy[:, [1, 3, 5, 7]]
		xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
		
		# # apply angle-based reduction of bounding boxes
		# radians = a * math.pi / 180
		# reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
		# x = (xy[:, 2] + xy[:, 0]) / 2
		# y = (xy[:, 3] + xy[:, 1]) / 2
		# w = (xy[:, 2] - xy[:, 0]) * reduction
		# h = (xy[:, 3] - xy[:, 1]) * reduction
		# xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
		
		# clip boxes
		xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
		xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
		
		# NOTE: Filter candidates
		i               = bbox_candidates(bbox_labels_new[:, 2:6].T * s, xy.T)
		bbox_labels_new = bbox_labels_new[i]
		bbox_labels_new[:, 2:6] = xy[i]
	
	return image_new, bbox_labels_new


@dispatch(np.ndarray, (int, float), (int, float))
def shift_bbox(
	xyxy: np.ndarray,
	ver : Union[int, float],
	hor : Union[int, float]
) -> np.ndarray:
	"""Shift the bounding box with the given `ver` and `hor` values.

	Args:
		xyxy (np.ndarray):
			The bounding box as [x1, y1, x2, y2].
		ver (int, float):
			The vertical value to shift.
		hor (int, float):
			The horizontal value to shift.

	Returns:
		xyxy_shifted (np.ndarray):
			The shifted bounding box.
	"""
	xyxy_shift       = xyxy.copy()
	xyxy_shift[:, 0] = xyxy[:, 0] + hor  # pad width
	xyxy_shift[:, 1] = xyxy[:, 1] + ver  # pad height
	xyxy_shift[:, 2] = xyxy[:, 2] + hor
	xyxy_shift[:, 3] = xyxy[:, 3] + ver
	return xyxy_shift


@dispatch(torch.Tensor, (int, float), (int, float))
def shift_bbox(
	xyxy: torch.Tensor,
	ver : Union[int, float],
	hor : Union[int, float]
) -> np.ndarray:
	"""Shift the bounding box with the given `ver` and `hor` values.

	Args:
		xyxy (torch.Tensor):
			The bounding box as [x1, y1, x2, y2].
		ver (int, float):
			The vertical value to shift.
		hor (int, float):
			The horizontal value to shift.

	Returns:
		xyxy_shifted (np.ndarray):
			The shifted bounding box.
	"""
	xyxy_shift       = xyxy.clone()
	xyxy_shift[:, 0] = xyxy[:, 0] + hor  # pad width
	xyxy_shift[:, 1] = xyxy[:, 1] + ver  # pad height
	xyxy_shift[:, 2] = xyxy[:, 2] + hor
	xyxy_shift[:, 3] = xyxy[:, 3] + ver
	return xyxy_shift


# MARK: - Box Formats Conversion
"""The coordination of bounding box's points.

(0, 0)              Image
	  ---------------------------------- -> columns
	  |                                |
	  |        ----- -> x              |
	  |        |   |                   |
	  |        |   |                   |
	  |        -----                   |
	  |        |                       |
	  |        V                       |
	  |        y                       |
	  ----------------------------------
	  |                                 (n, m)
	  V
	 rows
"""


@dispatch(np.ndarray)
def bbox_cxcyar_cxcyrh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = np.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = cxcyar[3]
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = cxcyar[:, 3]
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_cxcyar_cxcyrh(cxcyar: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = torch.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = cxcyar[3]
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = cxcyar[:, 3]
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcyar_cxcywh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, width, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(np.float32)
	if cxcyrh.ndim == 1:
		width     = np.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_cxcyar_cxcywh(cxcyar: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x, center_y, width, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = torch.sqrt(cxcyar[2] * cxcyar[3])
		height    = cxcyar[2] / width
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights      = cxcyar[:, 2] / widths
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcyar_cxcywh_norm(
	cxcyar: np.ndarray,
	height: Union[int, float],
	width : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
		- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcyar.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		w = np.sqrt(cxcyar[2] * cxcyar[3])
		h = cxcyar[2] / w
		cxcyrh[0] = cxcyrh[0] / width
		cxcyrh[1] = cxcyrh[1] / height
		cxcyrh[2] = (w / width)
		cxcyrh[3] = (h / width)
	elif cxcyrh.ndim == 2:
		ws = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		hs = cxcyar[:, 2] / ws
		cxcyrh[:, 0] = cxcyrh[:, 0] / width
		cxcyrh[:, 1] = cxcyrh[:, 1] / height
		cxcyrh[:, 2] = (ws / width)
		cxcyrh[:, 3] = (hs / width)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcyar_cxcywh_norm(
	cxcyar: torch.Tensor,
	height: Union[int, float],
	width : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
		- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcyar.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		w         = torch.sqrt(cxcyar[2] * cxcyar[3])
		h         = cxcyar[2] / w
		cxcyrh[0] = cxcyrh[0] / width
		cxcyrh[1] = cxcyrh[1] / height
		cxcyrh[2] = (w / width)
		cxcyrh[3] = (h / width)
	elif cxcyrh.ndim == 2:
		ws           = torch.sqrt(cxcyar[:, 2] * cxcyar[: , 3])
		hs           = cxcyar[:, 2] / ws
		cxcyrh[:, 0] = cxcyrh[:, 0] / width
		cxcyrh[:, 1] = cxcyrh[:, 1] / height
		cxcyrh[:, 2] = (ws / width)
		cxcyrh[:, 3] = (hs / width)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcyar_xywh(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, width, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyar.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		width   = np.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xywh[0] = xywh[0] - (width / 2.0)
		xywh[1] = xywh[1] - (height / 2.0)
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xywh[:, 0] = xywh[:, 0] - (widths / 2.0)
		xywh[:, 1] = xywh[:, 1] - (heights / 2.0)
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(torch.Tensor)
def bbox_cxcyar_xywh(cxcyar: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, width, height].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyar.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		width   = torch.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xywh[0] = xywh[0] - (width / 2.0)
		xywh[1] = xywh[1] - (height / 2.0)
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xywh[:, 0] = xywh[:, 0] - (widths / 2.0)
		xywh[:, 1] = xywh[:, 1] - (heights / 2.0)
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcyar_xyxy(cxcyar: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyar.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		width   = np.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xyxy[0] = xyxy[0] - (width / 2.0)
		xyxy[1] = xyxy[1] - (height / 2.0)
		xyxy[2] = xyxy[2] + (width / 2.0)
		xyxy[3] = xyxy[3] + (height / 2.0)
	elif xyxy.ndim == 2:
		widths     = np.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xyxy[:, 0] = xyxy[:, 0] - (widths / 2.0)
		xyxy[:, 1] = xyxy[:, 1] - (heights / 2.0)
		xyxy[:, 2] = xyxy[:, 2] + (widths / 2.0)
		xyxy[:, 3] = xyxy[:, 3] + (heights / 2.0)
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(torch.Tensor)
def bbox_cxcyar_xyxy(cxcyar: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, area, aspect_ratio] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyar.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		width   = torch.sqrt(cxcyar[2] * cxcyar[3])
		height  = cxcyar[2] / width
		xyxy[0] = xyxy[0] - (width / 2.0)
		xyxy[1] = xyxy[1] - (height / 2.0)
		xyxy[2] = xyxy[2] + (width / 2.0)
		xyxy[3] = xyxy[3] + (height / 2.0)
	elif xyxy.ndim == 2:
		widths     = torch.sqrt(cxcyar[:, 2] * cxcyar[:, 3])
		heights    = cxcyar[:, 2] / widths
		xyxy[:, 0] = xyxy[:, 0] - (widths / 2.0)
		xyxy[:, 1] = xyxy[:, 1] - (heights / 2.0)
		xyxy[:, 2] = xyxy[:, 2] + (widths / 2.0)
		xyxy[:, 3] = xyxy[:, 3] + (heights / 2.0)
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_cxcyrh_cxcyar(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyar = cxcyrh.copy()
	cxcyar = cxcyar.astype(float)
	if cxcyar.ndim == 1:
		width     = cxcyrh[2] * cxcyrh[3]
		height    = cxcyrh[3]
		cxcyar[2] = width * height
		cxcyar[3] = width / height
	elif cxcyar.ndim == 2:
		widths       = cxcyrh[:, 2] * cxcyrh[:, 3]
		heights      = cxcyrh[:, 3]
		cxcyar[:, 2] = widths * heights
		cxcyar[:, 3] = widths / heights
	else:
		raise ValueError(f"The array dimensions {cxcyar.ndim} is not "
						 f"supported.")
	return cxcyar


@dispatch(torch.Tensor)
def bbox_cxcyrh_cxcyar(cxcyrh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `area` is `width * height`.
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyar = cxcyrh.clone()
	cxcyar = cxcyar.float()
	if cxcyar.ndim == 1:
		width     = cxcyrh[2] * cxcyrh[3]
		height    = cxcyrh[3]
		cxcyar[2] = width * height
		cxcyar[3] = width / height
	elif cxcyar.ndim == 2:
		widths       = cxcyrh[:, 2] * cxcyrh[:, 3]
		heights      = cxcyrh[:, 3]
		cxcyar[:, 2] = widths * heights
		cxcyar[:, 3] = widths / heights
	else:
		raise ValueError(f"The array dimensions {cxcyar.ndim} is not "
						 f"supported.")
	return cxcyar


@dispatch(np.ndarray)
def bbox_cxcyrh_cxcywh(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, width, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcywh = cxcyrh.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[2] = cxcywh[2] * cxcywh[3]
	elif cxcywh.ndim == 2:
		cxcywh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(torch.Tensor)
def bbox_cxcyrh_cxcywh(cxcyrh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x, center_y, width, height].
	Where:
	 	- The `aspect_ratio` is `width / height`.
	"""
	cxcywh = cxcyrh.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[2] = cxcywh[2] * cxcywh[3]
	elif cxcywh.ndim == 2:
		cxcywh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcyrh_cxcywh_norm(
	cxcyrh: np.ndarray,
	height: Union[int, float],
	width : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_cxcyrh_cxcywh(cxcyrh)
	if cxcyrh.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcyrh.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcyrh_cxcywh_norm(
	cxcyrh: torch.Tensor,
	height: Union[int, float],
	width : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_cxcyrh_cxcywh(cxcyrh)
	if cxcyrh.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcyrh.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray)
def bbox_cxcyrh_xywh(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, width, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyrh.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		width   = xywh[2] * xywh[3]
		height  = xywh[3]
		xywh[0] = xywh[0] - width / 2.0
		xywh[1] = xywh[1] - height / 2.0
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = xywh[:, 2] * xywh[:, 3]
		heights    = xywh[:, 3]
		xywh[:, 0] = xywh[:, 0] - widths  / 2.0
		xywh[:, 1] = xywh[:, 1] - heights / 2.0
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(torch.Tensor)
def bbox_cxcyrh_xywh(cxcyrh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, width, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	xywh = cxcyrh.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		width   = xywh[2] * xywh[3]
		height  = xywh[3]
		xywh[0] = xywh[0] - width / 2.0
		xywh[1] = xywh[1] - height / 2.0
		xywh[2] = width
		xywh[3] = height
	elif xywh.ndim == 2:
		widths     = xywh[:, 2] * xywh[:, 3]
		heights    = xywh[:, 3]
		xywh[:, 0] = xywh[:, 0] - widths / 2.0
		xywh[:, 1] = xywh[:, 1] - heights / 2.0
		xywh[:, 2] = widths
		xywh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcyrh_xyxy(cxcyrh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyrh.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		half_height = xyxy[3] / 2.0
		half_width  = (xyxy[2] * xyxy[3]) / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_heights = xyxy[:, 3] / 2.0
		half_widths  = (xyxy[:, 2] * xyxy[:, 3]) / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(torch.Tensor)
def bbox_cxcyrh_xyxy(cxcyrh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, aspect_ratio, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y],
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	xyxy = cxcyrh.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		half_height = xyxy[3] / 2.0
		half_width  = (xyxy[2] * xyxy[3]) / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_heights = xyxy[:, 3] / 2.0
		half_widths  = (xyxy[:, 2] * xyxy[:, 3]) / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_cxcywh_cxcyar(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcywh[2] * cxcywh[3]
		cxcyrh[3] = cxcywh[2] / cxcywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
		cxcyrh[:, 3] = cxcywh[:, 2] / cxcywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_cxcywh_cxcyar(cxcywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcywh[2] * cxcywh[3]
		cxcyrh[3] = cxcywh[2] / cxcywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcywh[:, 2] * cxcywh[:, 3]
		cxcyrh[:, 3] = cxcywh[:, 2] / cxcywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_cxcywh_cxcyrh(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, aspect_ratio, height]
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_cxcywh_cxcyrh(cxcywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x, center_y, aspect_ratio, height]
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = cxcywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_cxcywh_norm(
	cxcywh: np.ndarray,
	height: Union[int, float],
	width : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = cxcywh.copy()
	cxcywh_norm = cxcywh_norm.astype(float)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_cxcywh_norm(
	cxcywh: torch.Tensor,
	height: Union[int, float],
	width : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = cxcywh.clone()
	cxcywh_norm = cxcywh_norm.float()
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray)
def bbox_cxcywh_xywh(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = cxcywh.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[0] = xywh[0] - xywh[2] / 2.0
		xywh[1] = xywh[1] - xywh[3] / 2.0
	elif xywh.ndim == 2:
		xywh[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
		xywh[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(torch.Tensor)
def bbox_cxcywh_xywh(cxcywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = cxcywh.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[0] = xywh[0] - xywh[2] / 2.0
		xywh[1] = xywh[1] - xywh[3] / 2.0
	elif xywh.ndim == 2:
		xywh[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
		xywh[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_cxcywh_xyxy(cxcywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = cxcywh.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		half_width  = xyxy[2] / 2.0
		half_height = xyxy[3] / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_widths  = xyxy[:, 2] / 2.0
		half_heights = xyxy[:, 3] / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(torch.Tensor)
def bbox_cxcywh_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x, center_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = cxcywh.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		half_width  = xyxy[2] / 2.0
		half_height = xyxy[3] / 2.0
		xyxy[3]     = xyxy[3] + half_height
		xyxy[2]     = xyxy[2] + half_width
		xyxy[1]     = xyxy[1] - half_height
		xyxy[0]     = xyxy[0] - half_width
	elif xyxy.ndim == 2:
		half_widths  = xyxy[:, 2] / 2.0
		half_heights = xyxy[:, 3] / 2.0
		xyxy[:, 3]   = xyxy[:, 3] + half_heights
		xyxy[:, 2]   = xyxy[:, 2] + half_widths
		xyxy[:, 1]   = xyxy[:, 1] - half_heights
		xyxy[:, 0]   = xyxy[:, 0] - half_widths
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_cxcyar(
	cxcywh_norm: np.ndarray,
	height     : Union[int, float],
	width      : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] * width
		cxcyrh[1] = cxcyrh[1] * height
		cxcyrh[2] = (cxcywh_norm[2] * width) * (cxcywh_norm[3] * height)
		cxcyrh[3] = (cxcywh_norm[2] * width) / (cxcywh_norm[3] * height)
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 2] = (cxcywh_norm[:, 2] * width) * (cxcywh_norm[:, 3] * height)
		cxcyrh[:, 3] = (cxcywh_norm[:, 2] * width) / (cxcywh_norm[:, 3] * height)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_cxcyar(
	cxcywh_norm: torch.Tensor,
	height	   : Union[int, float],
	width 	   : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] * width
		cxcyrh[1] = cxcyrh[1] * height
		cxcyrh[2] = (cxcywh_norm[2] * width) * (cxcywh_norm[3] * height)
		cxcyrh[3] = (cxcywh_norm[2] * width) / (cxcywh_norm[3] * height)
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 2] = (cxcywh_norm[:, 2] * width) * (cxcywh_norm[:, 3] * height)
		cxcyrh[:, 3] = (cxcywh_norm[:, 2] * width) / (cxcywh_norm[:, 3] * height)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_cxcyrh(
	cxcywh_norm: np.ndarray,
	height     : Union[int, float],
	width      : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] * width
		cxcyrh[1] = cxcyrh[1] * height
		cxcyrh[3] = cxcyrh[3] * height
		cxcyrh[2] = (cxcyrh[2] * width) / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 3] = cxcyrh[:, 3] * height
		cxcyrh[:, 2] = (cxcyrh[:, 2] * width) / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_cxcyrh(
	cxcywh_norm: torch.Tensor,
	height     : Union[int, float],
	width      : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcyrh = cxcywh_norm.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] * width
		cxcyrh[1] = cxcyrh[1] * height
		cxcyrh[3] = cxcyrh[3] * height
		cxcyrh[2] = (cxcyrh[2] * width) / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] * width
		cxcyrh[:, 1] = cxcyrh[:, 1] * height
		cxcyrh[:, 3] = cxcyrh[:, 3] * height
		cxcyrh[:, 2] = (cxcyrh[:, 2] * width) / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_cxcywh(
	cxcywh_norm: np.ndarray,
	height     : Union[int, float],
	width      : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, width, height].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh = cxcywh_norm.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[0] = cxcywh[0] * width
		cxcywh[1] = cxcywh[1] * height
		cxcywh[2] = cxcywh[2] * width
		cxcywh[3] = cxcywh[3] * height
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] * width
		cxcywh[:, 1] = cxcywh[:, 1] * height
		cxcywh[:, 2] = cxcywh[:, 2] * width
		cxcywh[:, 3] = cxcywh[:, 3] * height
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_cxcywh(
	cxcywh_norm: torch.Tensor,
	height     : Union[int, float],
	width      : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[center_x, center_y, width, height].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh = cxcywh_norm.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[0] = cxcywh[0] * width
		cxcywh[1] = cxcywh[1] * height
		cxcywh[2] = cxcywh[2] * width
		cxcywh[3] = cxcywh[3] * height
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] * width
		cxcywh[:, 1] = cxcywh[:, 1] * height
		cxcywh[:, 2] = cxcywh[:, 2] * width
		cxcywh[:, 3] = cxcywh[:, 3] * height
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_xywh(
	cxcywh_norm: np.ndarray,
	height     : Union[int, float],
	width      : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, width, height].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xywh = cxcywh_norm.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[3] = xywh[3] * height
		xywh[2] = xywh[2] * width
		xywh[1] = (xywh[1] * height) - (xywh[3] / 2.0)
		xywh[0] = (xywh[0] * width) - (xywh[2] / 2.0)
	elif xywh.ndim == 2:
		xywh[:, 3] = xywh[:, 3] * height
		xywh[:, 2] = xywh[:, 2] * width
		xywh[:, 1] = (xywh[:, 1] * height) - (xywh[:, 3] / 2.0)
		xywh[:, 0] = (xywh[:, 0] * width) - (xywh[:, 2] / 2.0)
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_xywh(
	cxcywh_norm: torch.Tensor,
	height     : Union[int, float],
	width      : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, width, height].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xywh = cxcywh_norm.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[3] = xywh[3] * height
		xywh[2] = xywh[2] * width
		xywh[1] = (xywh[1] * height) - (xywh[3] / 2.0)
		xywh[0] = (xywh[0] * width) - (xywh[2] / 2.0)
	elif xywh.ndim == 2:
		xywh[:, 3] = xywh[:, 3] * height
		xywh[:, 2] = xywh[:, 2] * width
		xywh[:, 1] = (xywh[:, 1] * height) - (xywh[:, 3] / 2.0)
		xywh[:, 0] = (xywh[:, 0] * width) - (xywh[:, 2] / 2.0)
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not "
						 f"supported.")
	return xywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_cxcywh_norm_xyxy(
	cxcywh_norm: np.ndarray,
	height     : Union[int, float],
	width      : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xyxy = cxcywh_norm.copy()
	xyxy = xyxy.astype(float)
	if xyxy.ndim == 1:
		xyxy[0] = width  * (cxcywh_norm[0] - cxcywh_norm[2] / 2)
		xyxy[1] = height * (cxcywh_norm[1] - cxcywh_norm[3] / 2)
		xyxy[2] = width  * (cxcywh_norm[0] + cxcywh_norm[2] / 2)
		xyxy[3] = height * (cxcywh_norm[1] + cxcywh_norm[3] / 2)
	elif xyxy.ndim == 2:
		xyxy[:, 0] = width  * (cxcywh_norm[:, 0] - cxcywh_norm[:, 2] / 2)
		xyxy[:, 1] = height * (cxcywh_norm[:, 1] - cxcywh_norm[:, 3] / 2)
		xyxy[:, 2] = width  * (cxcywh_norm[:, 0] + cxcywh_norm[:, 2] / 2)
		xyxy[:, 3] = height * (cxcywh_norm[:, 1] + cxcywh_norm[:, 3] / 2)
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_cxcywh_norm_xyxy(
	cxcywh_norm: torch.Tensor,
	height     : Union[int, float],
	width      : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[center_x_norm, center_y_norm, width_norm, height_norm] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	xyxy = cxcywh_norm.clone()
	xyxy = xyxy.float()
	if xyxy.ndim == 1:
		xyxy[0] = width  * (cxcywh_norm[0] - cxcywh_norm[2] / 2)
		xyxy[1] = height * (cxcywh_norm[1] - cxcywh_norm[3] / 2)
		xyxy[2] = width  * (cxcywh_norm[0] + cxcywh_norm[2] / 2)
		xyxy[3] = height * (cxcywh_norm[1] + cxcywh_norm[3] / 2)
	elif xyxy.ndim == 2:
		xyxy[:, 0] = width  * (cxcywh_norm[:, 0] - cxcywh_norm[:, 2] / 2)
		xyxy[:, 1] = height * (cxcywh_norm[:, 1] - cxcywh_norm[:, 3] / 2)
		xyxy[:, 2] = width  * (cxcywh_norm[:, 0] + cxcywh_norm[:, 2] / 2)
		xyxy[:, 3] = height * (cxcywh_norm[:, 1] + cxcywh_norm[:, 3] / 2)
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not "
						 f"supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_xywh_cxcyar(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, area, aspect_ratio].
	Where:
	 	- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = xywh[2] * xywh[3]
		cxcyrh[3] = xywh[2] / xywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = xywh[:, 2] * xywh[:, 3]
		cxcyrh[:, 3] = xywh[:, 2] / xywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_xywh_cxcyar(xywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, area, aspect_ratio]
	Where:
	 	- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = xywh[2] * xywh[3]
		cxcyrh[3] = xywh[2] / xywh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = xywh[:, 2] * xywh[:, 3]
		cxcyrh[:, 3] = xywh[:, 2] / xywh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xywh_cxcyrh(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, aspect_ratio, height].
	Where:
	 	- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_xywh_cxcyrh(xywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, aspect_ratio, height].
	Where:
	 	- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xywh.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		cxcyrh[0] = cxcyrh[0] + (cxcyrh[2] / 2.0)
		cxcyrh[1] = cxcyrh[1] + (cxcyrh[3] / 2.0)
		cxcyrh[2] = cxcyrh[2] / cxcyrh[3]
	elif cxcyrh.ndim == 2:
		cxcyrh[:, 0] = cxcyrh[:, 0] + (cxcyrh[:, 2] / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (cxcyrh[:, 3] / 2.0)
		cxcyrh[:, 2] = cxcyrh[:, 2] / cxcyrh[:, 3]
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xywh_cxcywh(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, width, height].
	"""
	cxcywh = xywh.copy()
	cxcywh = cxcywh.astype(float)
	if cxcywh.ndim == 1:
		cxcywh[0]    = cxcywh[0] + (cxcywh[2] / 2.0)
		cxcywh[1]    = cxcywh[1] + (cxcywh[3] / 2.0)
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] + (cxcywh[:, 2] / 2.0)
		cxcywh[:, 1] = cxcywh[:, 1] + (cxcywh[:, 3] / 2.0)
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(torch.Tensor)
def bbox_xywh_cxcywh(xywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x, center_y, width, height].
	"""
	cxcywh = xywh.clone()
	cxcywh = cxcywh.float()
	if cxcywh.ndim == 1:
		cxcywh[0]    = cxcywh[0] + (cxcywh[2] / 2.0)
		cxcywh[1]    = cxcywh[1] + (cxcywh[3] / 2.0)
	elif cxcywh.ndim == 2:
		cxcywh[:, 0] = cxcywh[:, 0] + (cxcywh[:, 2] / 2.0)
		cxcywh[:, 1] = cxcywh[:, 1] + (cxcywh[:, 3] / 2.0)
	else:
		raise ValueError(f"The array dimensions {cxcywh.ndim} is not "
						 f"supported.")
	return cxcywh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_xywh_cxcywh_norm(
	xywh  : np.ndarray,
	height: Union[int, float],
	width : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xywh_cxcywh(xywh)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_xywh_cxcywh_norm(
	xywh  : torch.Tensor,
	height: Union[int, float],
	width : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xywh_cxcywh(xywh)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray)
def bbox_xywh_xyxy(xywh: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = xywh.copy()
	if xyxy.ndim == 1:
		xyxy[2] = xyxy[2] + xyxy[0]
		xyxy[3] = xyxy[3] + xyxy[1]
	elif xyxy.ndim == 2:
		xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0]
		xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1]
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(torch.Tensor)
def bbox_xywh_xyxy(xywh: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, width, height] to
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyxy = xywh.clone()
	if xyxy.ndim == 1:
		xyxy[2] = xyxy[2] + xyxy[0]
		xyxy[3] = xyxy[3] + xyxy[1]
	elif xyxy.ndim == 2:
		xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0]
		xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1]
	else:
		raise ValueError(f"The array dimensions {xyxy.ndim} is not supported.")
	return xyxy


@dispatch(np.ndarray)
def bbox_xyxy_cxcyar(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width * height)
		cxcyrh[3] = (width / height)
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths * heights)
		cxcyrh[:, 3] = (widths / heights)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_xyxy_cxcyar(xyxy: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, area, aspect_ratio].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width * height)
		cxcyrh[3] = (width / height)
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths * heights)
		cxcyrh[:, 3] = (widths / heights)
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xyxy_cxcyrh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width / height)
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths / heights)
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_xyxy_cxcyrh(xyxy: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, aspect_ratio, height].
	Where:
		- The `aspect_ratio` is `width / height`.
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = (width / height)
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = (widths / heights)
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray)
def bbox_xyxy_cxcywh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, width, height].
	"""
	cxcyrh = xyxy.copy()
	cxcyrh = cxcyrh.astype(float)
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(torch.Tensor)
def bbox_xyxy_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x, center_y, width, height].
	"""
	cxcyrh = xyxy.clone()
	cxcyrh = cxcyrh.float()
	if cxcyrh.ndim == 1:
		width     = xyxy[2] - xyxy[0]
		height    = xyxy[3] - xyxy[1]
		cxcyrh[0] = cxcyrh[0] + (width / 2.0)
		cxcyrh[1] = cxcyrh[1] + (height / 2.0)
		cxcyrh[2] = width
		cxcyrh[3] = height
	elif cxcyrh.ndim == 2:
		widths       = xyxy[:, 2] - xyxy[:, 0]
		heights      = xyxy[:, 3] - xyxy[:, 1]
		cxcyrh[:, 0] = cxcyrh[:, 0] + (widths / 2.0)
		cxcyrh[:, 1] = cxcyrh[:, 1] + (heights / 2.0)
		cxcyrh[:, 2] = widths
		cxcyrh[:, 3] = heights
	else:
		raise ValueError(f"The array dimensions {cxcyrh.ndim} is not "
						 f"supported.")
	return cxcyrh


@dispatch(np.ndarray, (int, float), (int, float))
def bbox_xyxy_cxcywh_norm(
	xyxy  : np.ndarray,
	height: Union[int, float],
	width : Union[int, float]
) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xyxy_cxcywh(xyxy)
	cxcywh_norm = cxcywh_norm.astype(float)
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(torch.Tensor, (int, float), (int, float))
def bbox_xyxy_cxcywh_norm(
	xyxy  : torch.Tensor,
	height: Union[int, float],
	width : Union[int, float]
) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[center_x_norm, center_y_norm, width_norm, height_norm].
	Where:
	 	- The [center_x_norm, center_y_norm, width_norm, height_norm] are
		  normalized in the range `[0.0, 1.0]`.
		  For example:
			  `x_norm = absolute_x / image_width`
			  `height_norm = absolute_height / image_height`.
	"""
	cxcywh_norm = bbox_xyxy_cxcywh(xyxy)
	cxcywh_norm = cxcywh_norm.float()
	if cxcywh_norm.ndim == 1:
		cxcywh_norm[0] = cxcywh_norm[0] / width
		cxcywh_norm[1] = cxcywh_norm[1] / height
		cxcywh_norm[2] = cxcywh_norm[2] / width
		cxcywh_norm[3] = cxcywh_norm[3] / height
	elif cxcywh_norm.ndim == 2:
		cxcywh_norm[:, 0] = cxcywh_norm[:, 0] / width
		cxcywh_norm[:, 1] = cxcywh_norm[:, 1] / height
		cxcywh_norm[:, 2] = cxcywh_norm[:, 2] / width
		cxcywh_norm[:, 3] = cxcywh_norm[:, 3] / height
	else:
		raise ValueError(f"The array dimensions {cxcywh_norm.ndim} is not "
						 f"supported.")
	return cxcywh_norm


@dispatch(np.ndarray)
def bbox_xyxy_xywh(xyxy: np.ndarray) -> np.ndarray:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = xyxy.copy()
	xywh = xywh.astype(float)
	if xywh.ndim == 1:
		xywh[2] = xywh[2] - xywh[0]
		xywh[3] = xywh[3] - xywh[1]
	elif xywh.ndim == 2:
		xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
		xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(torch.Tensor)
def bbox_xyxy_xywh(xyxy: torch.Tensor) -> torch.Tensor:
	"""Convert the bounding box's format from
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y] to
	[top_left_x, top_left_y, width, height].
	"""
	xywh = xyxy.clone()
	xywh = xywh.float()
	if xywh.ndim == 1:
		xywh[2] = xywh[2] - xywh[0]
		xywh[3] = xywh[3] - xywh[1]
	elif xywh.ndim == 2:
		xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
		xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
	else:
		raise ValueError(f"The array dimensions {xywh.ndim} is not supported.")
	return xywh


@dispatch(np.ndarray)
def bbox_xyxy_center(xyxy: np.ndarray) -> np.ndarray:
	"""Return the center of the box of format
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyah = bbox_xyxy_cxcywh(xyxy)
	xyah = xyah.astype(float)
	if xyah.ndim == 1:
		return xyah[0:2]
	elif xyah.ndim == 2:
		return xyah[:, 0:2]
	else:
		raise ValueError(f"The array dimensions {xyah.ndim} is not supported.")


@dispatch(torch.Tensor)
def bbox_xyxy_center(xyxy: torch.Tensor) -> torch.Tensor:
	"""Return the center of the box of format
	[top_left_x, top_left_y, bottom_right_x, bottom_right_y].
	"""
	xyah = bbox_xyxy_cxcywh(xyxy)
	xyah = xyah.float()
	if xyah.ndim == 1:
		return xyah[0:2]
	elif xyah.ndim == 2:
		return xyah[:, 0:2]
	else:
		raise ValueError(f"The array dimensions {xyah.ndim} is not supported.")


# MARK: - Drawing Functions

def _draw_bbox(
	image    : np.ndarray,
	labels   : np.ndarray,
	colors   : Optional[list] = None,
	thickness: int            = 5
) -> np.ndarray:
	"""Draw bounding box(es) on image. If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (np.ndarray):
			Can be a 4D batch of numpy array or a single image.
		labels (np.ndarray):
			The bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in [
			xyxy] format.
		colors (list, optional):
			The list of colors.
		thickness (int):
			The thickness of the bounding box border.

	Returns:
		image (np.ndarray):
			The image with drawn bounding boxes.
	"""
	# NOTE: Draw bbox
	image = np.ascontiguousarray(image, dtype=np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	for i, l in enumerate(labels):
		class_id    = int(l[1])
		start_point = l[2:4].astype(np.int)
		end_point   = l[4:6].astype(np.int)
		color		= (255, 255, 255)
		if isinstance(colors, (tuple, list)) and len(colors) >= class_id:
			color = colors[class_id]
		image = cv2.rectangle(image, start_point, end_point, color, thickness)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image


@dispatch(np.ndarray, np.ndarray, list, int)
def draw_bbox(
	image    : np.ndarray,
	labels   : np.ndarray,
	colors   : Optional[list] = None,
	thickness: int            = 1
):
	"""Draw bounding box(es) on image(s). If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (np.ndarray):
			Can be a 4D batch of numpy array or a single image.
		labels (np.ndarray):
			The bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in
			[xyxy] format.
		colors (list, optional):
			The list of colors.
		thickness (int):
			The thickness of the bounding box border.
	"""
	# NOTE: Convert to channel-last
	image = to_channel_last(image)
	
	# NOTE: Unnormalize image
	image = unnormalize_image(image)
	image = image.astype(np.uint8)
	
	# NOTE: If the images are of shape [CHW]
	if image.ndim == 3:
		return _draw_bbox(image, labels, colors, thickness)
	
	# NOTE: If the images are of shape [BCHW]
	if image.ndim == 4:
		imgs = []
		for i, img in enumerate(image):
			l = labels[labels[:, 0] == i]
			imgs.append(_draw_bbox(img, l, colors, thickness))
		imgs = np.stack(imgs, axis=0).astype(np.unit8)
		return imgs
	
	raise ValueError(f"Do not support image with ndim: {image.ndim}.")


@dispatch(torch.Tensor, torch.Tensor, list, int)
def draw_bbox(
	image    : torch.Tensor,
	labels   : torch.Tensor,
	colors   : Optional[list] = None,
	thickness: int            = 1
):
	"""Draw bounding box(es) on image(s). If given the `colors`, use the color
	index corresponding with the `class_id` of the labels.

	Args:
		image (torch.Tensor):
			Can be a 4D batch of torch.Tensor or a single image.
		labels (torch.Tensor):
			The bounding box labels where the bounding boxes coordinates are
			located at: labels[:, 2:6]. Also, the bounding boxes are in
			[xyxy] format.
		colors (list, optional):
			The list of colors.
		thickness (int):
			The thickness of the bounding box border.
	"""
	image_np  = image.numpy()
	labels_np = labels.numpy()
	return draw_bbox(image_np, labels_np, colors, thickness)
