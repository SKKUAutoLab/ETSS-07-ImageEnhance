#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the interface for ``torchdiffeq`` package.

References:
    - Code: https://github.com/rtqichen/torchdiffeq
    
Last updated: 2025-09-18
"""

__all__ = [
    "odeint",
    "odeint_adjoint",
    "odeint_dense",
    "odeint_event",
]

from .torchdiffeq import odeint, odeint_adjoint, odeint_dense, odeint_event

__version__ = "0.2.5"
