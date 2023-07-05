#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Style printing.
"""

from __future__ import annotations

from sty import bg, ef, fg, rs
from sty import RgbFg, Style


def prints(s: str):
    """Print normal status."""
    s = fg.li_green + s + fg.rs
    print(s)


def printw(s: str):
    """Print warning status."""
    s = fg.li_yellow + s + fg.rs
    print(s)


def printe(s: str):
    """Print error status."""
    s = fg.li_red + s + fg.rs
    print(s)
