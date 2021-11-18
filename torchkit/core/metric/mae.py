#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MAE evaluation metric
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


# MARK: - MAE

@dispatch(np.ndarray, np.ndarray)
def mae(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """"Calculate MAE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    y_hat = y_hat.astype("float64")
    y     = y.astype("float64")
    score = np.mean(np.abs(y_hat - y) ** 2)
    return score


@dispatch(torch.Tensor, torch.Tensor)
def mae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """"Calculate MAE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    y_hat = y_hat.type(torch.float64)
    y     = y.type(torch.float64)
    score = torch.mean(torch.abs(y_hat - y) ** 2)
    return score


# noinspection PyMethodMayBeStatic
@METRICS.register(name="mae")
class MAE(nn.Module):
    """Calculate MAE (Mean Absolute Error).

    Attributes:
        name (str):
            Name of the loss.
    """
    
    # MARK: Magic Functions

    def __init__(self):
        super().__init__()
        self.name = "mae"
    
    # MARK: Forward Pass
    
    def forward(
        self,
        y_hat: Union[torch.Tensor, np.ndarray],
        y    : Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        return mae(y_hat=y_hat, y=y)
