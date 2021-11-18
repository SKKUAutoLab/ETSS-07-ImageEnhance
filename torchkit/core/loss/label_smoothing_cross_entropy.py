#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cross Entropy w/ smoothing
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES

logger = logging.getLogger()


# MARK: - LabelSmoothingCrossEntropy

@LOSSES.register(name="label_smoothing_cross_entropy")
class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing."""
    
    # MARK: Magic Functions
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
    
    # MARK: Forward Pass
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logprobs 	= F.log_softmax(y_hat, dim=-1)
        nll_loss 	= -logprobs.gather(dim=-1, index=y.unsqueeze(1))
        nll_loss	= nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss 		= self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
