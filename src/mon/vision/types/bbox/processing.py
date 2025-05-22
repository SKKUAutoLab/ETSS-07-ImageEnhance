#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements bounding box manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
    - Transformations.
"""

__all__ = [
    "bbox_center_distance",
    "bbox_ciou",
    "bbox_coco_to_voc",
    "bbox_coco_to_yolo",
    "bbox_cxcywhn_to_xywh",
    "bbox_cxcywhn_to_xyxy",
    "bbox_cxcywhn_to_xyxyn",
    "bbox_diou",
    "bbox_giou",
    "bbox_iou",
    "bbox_voc_to_coco",
    "bbox_voc_to_yolo",
    "bbox_xywh_to_cxcywhn",
    "bbox_xywh_to_xyxy",
    "bbox_xywh_to_xyxyn",
    "bbox_xyxy_to_cxcywhn",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xyxyn",
    "bbox_xyxyn_to_cxcywhn",
    "bbox_xyxyn_to_xywh",
    "bbox_xyxyn_to_xyxy",
    "bbox_yolo_to_coco",
    "bbox_yolo_to_voc",
    "convert_bbox",
    "split_image_and_bboxes",
]

import math

import numpy as np

from mon.constants import ShapeCode
from mon.vision.types import image as I
from mon.vision.types.bbox import utils


# ----- Split -----
def split_image_and_bboxes(
    image : np.ndarray,
    bboxes: np.ndarray,
    n     : int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split an image into ``n`` equal parts and adjust YOLO-format bboxes accordingly.

    Args:
        image: Image as ``numpy.ndarray`` [H, W, C].
        bboxes: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        n: Number of parts to split into (positive integer). Default is ``2``.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, bboxes, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"[image] must be a 3D numpy array [H, W, C], got {image.shape}.")
    if not isinstance(bboxes, np.ndarray) or bboxes.ndim != 2 or bboxes.shape[1] < 4:
        raise ValueError(f"[bboxes] must be a 2D numpy array [N, M] with M >= 4, got {bboxes.shape}.")
    if n < 1:
        raise ValueError(f"[n] must be a positive integer, got {n}.")

    h, w = I.image_size(image)
    if n > h * w:
        raise ValueError(f"[n] ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []
    sub_bboxes = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

            # Adjust bboxes
            sub_bboxes_i     = []
            sub_h_i, sub_w_i = sub_image.shape[:2]
            for bbox in bboxes:
                cx_n, cy_n, w_n, h_n = bbox[:4]
                cx = cx_n * w
                cy = cy_n * h
                x1 = cx - w_n * w / 2
                x2 = cx + w_n * w / 2
                y1 = cy - h_n * h / 2
                y2 = cy + h_n * h / 2

                # Check if bbox intersects sub-image
                if x2 > x_start and x1 < x_end and y2 > y_start and y1 < y_end:
                    # Compute new bbox in sub-image coordinates
                    x1_new   = max(x1, x_start) - x_start
                    x2_new   = min(x2, x_end)   - x_start
                    y1_new   = max(y1, y_start) - y_start
                    y2_new   = min(y2, y_end)   - y_start
                    cx_n_new = (x1_new + x2_new) / 2 / sub_w_i
                    cy_n_new = (y1_new + y2_new) / 2 / sub_h_i
                    w_n_new  = (x2_new - x1_new) / sub_w_i
                    h_n_new  = (y2_new - y1_new) / sub_h_i
                    if w_n_new > 0 and h_n_new > 0:
                        bbox_new = np.concatenate(([cx_n_new, cy_n_new, w_n_new, h_n_new], bbox[4:]))
                        sub_bboxes_i.append(bbox_new)
            sub_bboxes.append(np.array(sub_bboxes_i) if sub_bboxes_i else np.zeros((0, bboxes.shape[1]), dtype=np.float32))

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))
        sub_bboxes.append(np.zeros((0, bboxes.shape[1]), dtype=np.float32))

    return sub_images, sub_bboxes


# ----- Calculation -----
def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise IoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.
    """
    # Ensure 2D arrays
    bbox1 = utils.check_valid_bbox(bbox1)
    bbox2 = utils.check_valid_bbox(bbox2)

    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    
    # IoU calculation.
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    
    # Intersection area
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    # Union area
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
           + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    return iou


def bbox_giou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute generalized IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise GIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = utils.check_valid_bbox(bbox1)
    bbox2 = utils.check_valid_bbox(bbox2)
    
    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Enclosing box coordinates
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])

    # Enclosing area
    wc   = xxc2 - xxc1
    hc   = yyc2 - yyc1
    area_enclose = wc * hc

    # GIoU
    giou = iou - (area_enclose - union) / area_enclose
    # giou = (giou + 1.0) / 2.0  # Commented out: GIoU typically in [-1, 1], not [0, 1]
    return giou


def bbox_diou(bbox1: np.ndarray,  bbox2: np.ndarray) -> np.ndarray:
    """Compute distance IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise DIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = utils.check_valid_bbox(bbox1)
    bbox2 = utils.check_valid_bbox(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Center distances
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # DIoU
    diou = iou - inner_diag / outer_diag
    # diou = (diou + 1) / 2.0  # Commented: DIoU typically in [-1, 1], not [0, 1]
    return diou


def bbox_ciou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute complete IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise CIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = utils.check_valid_bbox(bbox1)
    bbox2 = utils.check_valid_bbox(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Center distances
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # Aspect ratio term
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    h2 += 1.0  # Prevent division by zero
    h1 += 1.0  # Prevent division by zero
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v      = (4 / (np.pi ** 2)) * (arctan ** 2)
    S      = 1 - iou
    alpha  = v / (S + v)

    # CIoU
    ciou = iou - inner_diag / outer_diag - alpha * v
    # ciou = (ciou + 1) / 2.0  # Commented: CIoU typically in [-1, 1], not [0, 1]
    return ciou


def bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Measure center distance(s) between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise center distances as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    Notes:
        Coarse implementation, not recommended alone for association due to instability.
    """
    # Ensure 2D arrays
    bbox1 = utils.check_valid_bbox(bbox1)
    bbox2 = utils.check_valid_bbox(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Center coordinates
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0  # Fixed: Use bbox1 only
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0  # Fixed: Use bbox1 only
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0  # Fixed: Use bbox2 only
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0  # Fixed: Use bbox2 only

    # Squared Euclidean distance
    ct_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Euclidean distance
    ct_dist = np.sqrt(ct_dist2)

    # Normalize and invert to [0, 1] (smaller distance = higher value)
    ct_dist_max = np.max(ct_dist)
    if ct_dist_max > 0:  # Avoid division by zero
        ct_dist = ct_dist / ct_dist_max
        ct_dist = ct_dist_max - ct_dist  # Invert: max distance = 0, min = max
    else:
        ct_dist = np.ones_like(ct_dist)  # All distances 0 -> all 1

    return ct_dist


# ----- Convert -----
def bbox_cxcywhn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
    """
    bbox = utils.check_valid_bbox(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    w = w_n * width
    h = h_n * height
    x = (cx_n * width)  - (w / 2.0)
    y = (cy_n * height) - (h / 2.0)
    # Combine processed columns with rest
    return np.stack([x, y, w, h] + rest, axis=-1)


def bbox_cxcywhn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
    """
    bbox = utils.check_valid_bbox(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    x1 = width  * (cx_n - w_n / 2)
    y1 = height * (cy_n - h_n / 2)
    x2 = width  * (cx_n + w_n / 2)
    y2 = height * (cy_n + h_n / 2)
    return np.stack([x1, y1, x2, y2] + rest, axis=-1)


def bbox_cxcywhn_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
    """
    bbox = utils.check_valid_bbox(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    x1 = (cx_n - w_n / 2)
    y1 = (cy_n - h_n / 2)
    x2 = (cx_n + w_n / 2)
    y2 = (cy_n + h_n / 2)
    return np.stack([x1, y1, x2, y2] + rest, axis=-1)


def bbox_xywh_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
    """
    bbox = utils.check_valid_bbox(bbox)
    x, y, w, h, *rest = bbox.T
    cx   = x + (w / 2.0)
    cy   = y + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack([cx_n, cy_n, w_n, h_n] + rest, axis=-1)


def bbox_xywh_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
    """
    bbox = utils.check_valid_bbox(bbox)
    x, y, w, h, *rest = bbox.T
    x2 = x + w
    y2 = y + h
    return np.stack([x, y, x2, y2] + rest, axis=-1)


def bbox_xywh_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
    """
    x, y, w, h, *rest = bbox.T
    x2   = x + w
    y2   = y + h
    x1_n = x / width
    y1_n = y / height
    x2_n = x2 / width
    y2_n = y2 / height
    return np.stack([x1_n, y1_n, x2_n, y2_n] + rest, axis=-1)


def bbox_xyxy_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
    """
    bbox = utils.check_valid_bbox(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    w    = x2 - x1
    h    = y2 - y1
    cx   = x1 + (w / 2.0)
    cy   = y1 + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack([cx_n, cy_n, w_n, h_n] + rest, axis=-1)


def bbox_xyxy_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYWH format, pixel coordinates.
    """
    x1, y1, x2, y2, *rest = bbox.T
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1, y1, w, h), axis=-1)


def bbox_xyxy_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
    """
    bbox = utils.check_valid_bbox(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    x1_n = x1 / width
    y1_n = y1 / height
    x2_n = x2 / width
    y2_n = y2 / height
    return np.stack([x1_n, y1_n, x2_n, y2_n] + rest, axis=-1)


def bbox_xyxyn_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXYN to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
    """
    x1, y1, x2, y2, *rest = bbox.T
    w_norm  = x2 - x1
    h_norm  = y2 - y1
    cx_norm = x1 + (w_norm / 2.0)
    cy_norm = y1 + (h_norm / 2.0)
    return np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)


def bbox_xyxyn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXYN to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
    """
    bbox = utils.check_valid_bbox(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    x1 = x1 * width
    x2 = x2 * width
    y1 = y1 * height
    y2 = y2 * height
    w  = x2 - x1
    h  = y2 - y1
    return np.stack([x1, y1, w, h] + rest, axis=-1)


def bbox_xyxyn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXYN to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
    """
    bbox = utils.check_valid_bbox(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    x1 = x1 * width
    x2 = x2 * width
    y1 = y1 * height
    y2 = y2 * height
    return np.stack([x1, y1, x2, y2] + rest, axis=-1)


bbox_coco_to_voc  = bbox_xywh_to_xyxy
bbox_coco_to_yolo = bbox_xywh_to_cxcywhn
bbox_voc_to_coco  = bbox_xyxy_to_xywh
bbox_voc_to_yolo  = bbox_xyxy_to_cxcywhn
bbox_yolo_to_coco = bbox_cxcywhn_to_xywh
bbox_yolo_to_voc  = bbox_cxcywhn_to_xyxy


def convert_bbox(bbox: np.ndarray, code: ShapeCode | int, height: int, width: int) -> np.ndarray:
    """Convert bounding box between formats.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], input format varies by code.
        code: Conversion code as ``ShapeCode`` or ``int``.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], output format varies by code.

    Raises:
        ValueError: If ``code`` is invalid.
    """
    code = ShapeCode.from_value(value=code)
    match code:
        case ShapeCode.SAME:
            return bbox
        case ShapeCode.VOC2COCO | ShapeCode.XYXY2XYWH:
            return bbox_voc_to_coco(bbox, height, width)
        case ShapeCode.VOC2YOLO | ShapeCode.XYXY2CXCYN:
            return bbox_voc_to_yolo(bbox, height, width)
        case ShapeCode.COCO2VOC | ShapeCode.XYWH2XYXY:
            return bbox_coco_to_voc(bbox, height, width)
        case ShapeCode.COCO2YOLO | ShapeCode.XYWH2CXCYN:
            return bbox_coco_to_yolo(bbox, height, width)
        case ShapeCode.YOLO2VOC | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_voc(bbox, height, width)
        case ShapeCode.YOLO2COCO | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_coco(bbox, height, width)
        case _:
            raise ValueError(f"[code] is invalid: {code}.")
