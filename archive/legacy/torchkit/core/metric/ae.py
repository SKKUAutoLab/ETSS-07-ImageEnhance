#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AE evaluation metric.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn

from .builder import METRICS

logger = logging.getLogger()


# MARK: - AE

@dispatch(np.ndarray, np.ndarray)
def ae(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """"Calculate AE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    y_hat = y_hat.astype("float64")
    y     = y.astype("float64")
    score = np.abs(np.mean(y_hat) - np.mean(y))
    return score


@dispatch(torch.Tensor, torch.Tensor)
def ae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """"Calculate AE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    y_hat = y_hat.type(torch.float64)
    y     = y.type(torch.float64)
    score = torch.abs(torch.mean(y_hat) - torch.mean(y))
    return score


# noinspection PyMethodMayBeStatic
@METRICS.register(name="ae")
class AE(nn.Module):
    """Calculate AE (Absolute Error).

    Attributes:
        name (str):
            Name of the loss.
    """
    
    # MARK: Magic Functions

    def __init__(self):
        super().__init__()
        self.name = "ae"
    
    # MARK: Forward Pass
    
    def forward(
        self,
        y_hat: Union[torch.Tensor, np.ndarray],
        y    : Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        return ae(y_hat=y_hat, y=y)
