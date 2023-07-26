#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for :mod:`mon.nn`.
"""

from __future__ import annotations

__all__ = [
    "check_kernel_size",
    "to_2d_kernel_size",
    "to_3d_kernel_size",
]

import mon.foundation
from mon.nn import _size_2_t, _size_3_t, _size_any_t

console = mon.foundation.console


# region Helper Functions

def check_kernel_size(
    kernel_size: _size_any_t,
    min_value  : int  = 0,
    allow_even : bool = False,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        assert isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value), \
            f"`kernel_size` must be an {fmt} integer bigger than {min_value}. " \
            f"Gotcha {size} on {kernel_size}."
        
        
def to_2d_kernel_size(kernel_size: _size_2_t) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2
        console.log(f"2D Kernel size should have a length of 2.")
        ky, kx = kernel_size
    ky = int(ky)
    kx = int(kx)
    return ky, kx


def to_3d_kernel_size(kernel_size: _size_3_t) -> tuple[int, int, int]:
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        assert len(kernel_size) == 3
        console.log(f"3D Kernel size should have a length of 3.")
        kz, ky, kx = kernel_size
    kz = int(kz)
    ky = int(ky)
    kx = int(kx)
    return kz, ky, kx

# endregion
