#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements linearity layers."""

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import Bilinear, Identity, LazyLinear, Linear
