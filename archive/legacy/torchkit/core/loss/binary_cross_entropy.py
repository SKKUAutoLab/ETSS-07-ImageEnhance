#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binary Cross Entropy w/ a few extras.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES

logger = logging.getLogger()


# MARK: - BinaryCrossEntropy

@LOSSES.register(name="binary_loss_entropy")
class BinaryCrossEntropy(nn.Module):
    """BCE with optional one-hot from dense targets, label smoothing,
    thresholding. NOTE for experiments comparing CE to BCE /w label
    smoothing, may remove.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        smoothing       : float                  = 0.1,
        target_threshold: Optional[float]        = None,
        weight          : Optional[torch.Tensor] = None,
        reduction       : str                    = "mean",
        pos_weight      : Optional[torch.Tensor] = None
    ):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing        = smoothing
        self.target_threshold = target_threshold
        self.reduction        = reduction
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    # MARK: Forward Pass
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert y_hat.shape[0] == y.shape[0]
        if y_hat.shape != y.shape:
            # NOTE currently assume smoothing or other label softening is
            # applied upstream if targets are already sparse
            num_classes = y_hat.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other
            # impl out there differ
            off_value = self.smoothing / num_classes
            on_value  = 1.0 - self.smoothing + off_value
            target    = y.long().view(-1, 1)
            target    = torch.full(
                (target.size()[0], num_classes), off_value,
                device=y_hat.device, dtype=y_hat.dtype
            ).scatter_(1, target, on_value)
        
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            y = y.gt(self.target_threshold).to(dtype=y.dtype)
       
        return F.binary_cross_entropy_with_logits(
            y_hat, y, self.weight, pos_weight=self.pos_weight,
            reduction=self.reduction
        )
