#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for points.
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np

logger = logging.getLogger()


# MARK: - Calculation

def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
	"""Calculate Euclidean distance between 2 points.
	"""
	return sqrt(((point_a[0] - point_b[0]) ** 2) +
				((point_a[1] - point_b[1]) ** 2))
