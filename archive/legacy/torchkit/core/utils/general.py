#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detector Head.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger()


def check_anchor_order(m):
    """Check anchor order against stride order for YOLO Detect() module m, and
    correct if necessary.
    """
    a  = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:]     = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
        

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor
