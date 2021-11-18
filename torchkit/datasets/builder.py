#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class for creating and registering datasets.
"""

from __future__ import annotations

from torchkit.core.factory import Factory

DATASETS    = Factory(name="dataset")
DATAMODULES = Factory(name="datamodule")
