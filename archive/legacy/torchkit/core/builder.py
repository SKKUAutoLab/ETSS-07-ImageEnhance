#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""All factory classes used in the `torchkit` package. We define them here to
avoid circular dependency.
"""

from __future__ import annotations

from .layer import ACT_LAYERS
from .layer import ATTN_LAYERS
from .layer import BOTTLENECK_LAYERS
from .layer import CONV_ACT_LAYERS
from .layer import CONV_LAYERS
from .layer import CONV_NORM_ACT_LAYERS
from .layer import DROP_LAYERS
from .layer import EMBED_LAYERS
from .layer import HEADS
from .layer import LINEAR_LAYERS
from .layer import MLP_LAYERS
from .layer import NORM_ACT_LAYERS
from .layer import NORM_LAYERS
from .layer import PADDING_LAYERS
from .layer import PLUGIN_LAYERS
from .layer import POOL_LAYERS
from .layer import SAMPLING_LAYERS
from .loss import LOSSES
from .metric import METRICS
from .optim import OPTIMIZERS
from .parallel import MODULE_WRAPPERS
from .runner import LOGGERS
from .scheduler import SCHEDULERS

__all__ = [
    "ACT_LAYERS",
    "ATTN_LAYERS",
    "BOTTLENECK_LAYERS",
    "CONV_LAYERS",
    "CONV_ACT_LAYERS",
    "CONV_NORM_ACT_LAYERS",
    "DROP_LAYERS",
    "EMBED_LAYERS",
    "HEADS",
    "LINEAR_LAYERS",
    "LOGGERS",
    "LOSSES",
    "METRICS",
    "MLP_LAYERS",
    "MODULE_WRAPPERS",
    "NORM_LAYERS",
    "NORM_ACT_LAYERS",
    "OPTIMIZERS",
    "PADDING_LAYERS",
    "PLUGIN_LAYERS",
    "POOL_LAYERS",
    "SAMPLING_LAYERS",
    "SCHEDULERS",
]
