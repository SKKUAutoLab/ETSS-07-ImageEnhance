#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RMSE evaluation metric.
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


# MARK: - RMSE

@dispatch(np.ndarray, np.ndarray)
def rmse(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """"Calculate RMSE (Root Mean Square Error) score between 2 4D-/3D-
    channel-first- images.
    """
    y_hat = y_hat.astype("float64")
    y     = y.astype("float64")
    score = np.sqrt(np.mean((y_hat - y) ** 2.0))
    return score


@dispatch(torch.Tensor, torch.Tensor)
def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """"Calculate RMSE (Root Mean Square Error) score between 2 4D-/3D-
    channel-first- images.
    """
    y_hat = y_hat.type(torch.float64)
    y     = y.type(torch.float64)
    score = torch.sqrt(torch.mean((y_hat - y) ** 2.0))
    return score


# noinspection PyMethodMayBeStatic
@METRICS.register(name="rmse")
class RMSE(nn.Module):
    """Calculate RMSE (Root Mean Square Error).

    Attributes:
        name (str, optional):
            Name of the loss. Default: `rmse`.
    """
    
    # MARK: Magic Functions

    def __init__(self):
        super().__init__()
        self.name = "rmse"
    
    # MARK: Forward Pass
    
    def forward(
        self,
        y_hat: Union[torch.Tensor, np.ndarray],
        y    : Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        return rmse(y_hat=y_hat, y=y)
