#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for bounding box.

Common Tasks:
    - Property accessors
    - Validation checks
    - Miscellaneous
"""

__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_corners",
    "bbox_corners_pts",
    "check_valid_bbox",
    "enclosing_bbox",
    "is_bbox_coco",
    "is_bbox_cxcywhn",
    "is_bbox_normalized",
    "is_bbox_voc",
    "is_bbox_xywh",
    "is_bbox_xyxy",
    "is_bbox_yolo",
]

import numpy as np


# ----- Access -----
def bbox_area(bbox: np.ndarray) -> np.ndarray:
    """Compute area of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Area(s) as ``np.ndarray`` in [1] or [N] shape.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = check_valid_bbox(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    return (x2 - x1) * (y2 - y1)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Compute center(s) of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Center(s) as ``np.ndarray`` in [1, 2] or [N, 2], [cx, cy] format.

    Raises:
        ValueError: If bbox is not 1D or 2D.
    """
    bbox = check_valid_bbox(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    cx   = x1 + (x2 - x1) / 2.0
    cy   = y1 + (y2 - y1) / 2.0
    return np.stack((cx, cy), -1)


def bbox_corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4+] or [N, 4+], XYXY format

    Returns:
        Corners as ``np.ndarray`` in [N, 8], [x1, y1, x2, y2, x3, y3, x4, y4] format

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D
    """
    bbox = check_valid_bbox(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.hstack((c_x1, c_y1, c_x2, c_y2, c_x3, c_y3, c_x4, c_y4))


def bbox_corners_pts(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es) as points.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Corners as ``np.ndarray`` in
        [N, 4, 2], [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] format.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = check_valid_bbox(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.array([[c_x1, c_y1], [c_x2, c_y2], [c_x3, c_y3], [c_x4, c_y4]], np.int32)


def enclosing_bbox(bbox: np.ndarray) -> np.ndarray:
    """Get enclosing box(es) for rotated corners.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [..., 8], [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        Box(es) as ``np.ndarray`` in [..., 4], XYXY format.

    Raises:
        ValueError: If bbox last dimension is not 8.
    """
    if bbox.shape[-1] < 8:
        raise ValueError(f"[bbox] last dimension must be 8, got {bbox.shape[-1]}.")
    x_ = bbox[:, [0, 2, 4, 6]]
    y_ = bbox[:, [1, 3, 5, 7]]
    x1 = np.min(x_, 1).reshape(-1, 1)
    y1 = np.min(y_, 1).reshape(-1, 1)
    x2 = np.max(x_, 1).reshape(-1, 1)
    y2 = np.max(y_, 1).reshape(-1, 1)
    return np.hstack((x1, y1, x2, y2, bbox[:, 8:]))


# ----- Validation Check -----
def check_valid_bbox(bbox: np.ndarray | list | tuple) -> np.ndarray:
    """Check if a bounding box is valid.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].

    Returns:
        ``True`` if valid, ``False`` otherwise.
    """
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox, dtype=np.float32)
    if not isinstance(bbox, np.ndarray):
        raise ValueError(f"[bbox] must be a numpy.ndarray, got {type(bbox)}.")
    if bbox.ndim == 1:
        bbox = bbox.reshape(1, -1)
    if bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"[bbox] must be a 2D array [N, M] with M >= 4 or a 1D array [M] with M >= 4, got {bbox.shape}.")
    #if not np.all((bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 2] > 0) & (bbox[:, 3] > 0)):
    #    raise ValueError(f"Invalid bbox values.")
    return bbox


def is_bbox_normalized(bbox: np.ndarray) -> bool:
    """Check if a bounding box is normalized to range [0.0, 1.0].

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].

    Returns:
        ``True`` if normalized, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    return np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))


def is_bbox_cxcywhn(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in CXCYWHN format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in CXCYWHN format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    return (
        np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))
        and np.all((bbox[:, 2:4] > 0))  # Width and height must be positive
    )


def is_bbox_xyxy(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in XYXY format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in XYXY format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    if is_bbox_cxcywhn(bbox, height, width):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w > x and h > y:  # VOC: x_max > x_min, y_max > y_min
        return True
    else:
        return False


def is_bbox_xywh(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in XYWH format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in XYWH format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    if is_bbox_cxcywhn(bbox, height, width):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w + x > x and h + y > y:  # VOC: w=x_max, h=y_max, so x_min+w > x_min
        return True  # COCO: w=width, h=height
    else:
        return False


is_bbox_coco = is_bbox_xywh
is_bbox_voc  = is_bbox_xyxy
is_bbox_yolo = is_bbox_cxcywhn
