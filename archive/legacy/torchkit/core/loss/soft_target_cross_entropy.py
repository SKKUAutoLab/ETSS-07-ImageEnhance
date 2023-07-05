#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cross Entropy w/ soft targets
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES

logger = logging.getLogger()


# MARK: - SoftTargetCrossEntropy

@LOSSES.register(name="soft_target_cross_entropy")
class SoftTargetCrossEntropy(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self):
        super().__init__()
    
    # MARK: Forward Pass
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-y * F.log_softmax(y_hat, dim=-1), dim=-1)
        return loss.mean()
