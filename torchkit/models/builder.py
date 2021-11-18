#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class for creating and registering deep learning models and
components.
"""

from __future__ import annotations

import logging

from torchkit.core.builder import ACT_LAYERS
from torchkit.core.builder import ATTN_LAYERS
from torchkit.core.builder import BOTTLENECK_LAYERS
from torchkit.core.builder import CONV_ACT_LAYERS
from torchkit.core.builder import CONV_LAYERS
from torchkit.core.builder import CONV_NORM_ACT_LAYERS
from torchkit.core.builder import DROP_LAYERS
from torchkit.core.builder import EMBED_LAYERS
from torchkit.core.builder import HEADS
from torchkit.core.builder import LINEAR_LAYERS
from torchkit.core.builder import LOGGERS
from torchkit.core.builder import LOSSES
from torchkit.core.builder import METRICS
from torchkit.core.builder import MLP_LAYERS
from torchkit.core.builder import MODULE_WRAPPERS
from torchkit.core.builder import NORM_ACT_LAYERS
from torchkit.core.builder import NORM_LAYERS
from torchkit.core.builder import OPTIMIZERS
from torchkit.core.builder import PADDING_LAYERS
from torchkit.core.builder import PLUGIN_LAYERS
from torchkit.core.builder import POOL_LAYERS
from torchkit.core.builder import SAMPLING_LAYERS
from torchkit.core.builder import SCHEDULERS
from torchkit.core.factory import Factory

logger = logging.getLogger()

__all__ = [
    "ACT_LAYERS",
    "ATTN_LAYERS",
	"BACKBONES",
    "BOTTLENECK_LAYERS",
	"CLASSIFIERS",
    "CONV_LAYERS",
    "CONV_ACT_LAYERS",
    "CONV_NORM_ACT_LAYERS",
	"DETECTORS",
    "DROP_LAYERS",
    "EMBED_LAYERS",
	"ENHANCERS",
	"HEADS",
    "LINEAR_LAYERS",
    "LOGGERS",
    "LOSSES",
    "METRICS",
	"MODELS",
    "MLP_LAYERS",
    "MODULE_WRAPPERS",
	"NECKS",
    "NORM_LAYERS",
    "NORM_ACT_LAYERS",
    "OPTIMIZERS",
    "PADDING_LAYERS",
    "PLUGIN_LAYERS",
    "POOL_LAYERS",
    "SAMPLING_LAYERS",
    "SCHEDULERS",
]

BACKBONES   = Factory(name="backbones")
CLASSIFIERS = Factory(name="classifiers")
DETECTORS   = Factory(name="detectors")
ENHANCERS   = Factory(name="enhancers")
MODELS 	    = Factory(name="models")
NECKS 	    = Factory(name="necks")
