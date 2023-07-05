#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Charbonnier Loss.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .builder import LOSSES

logger = logging.getLogger()


# MARK: - CharbonnierLoss

@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(nn.Module):
	"""Charbonnier Loss (L1).
	
	Attributes:
		eps (float):
			The eps value.
		name (str):
			Name of the loss.
	"""
	
	# MARK: Magic Functions

	def __init__(self, eps: float = 1e-3):
		super().__init__()
		self.eps  = eps
		self.name = "charbonnier_loss"
	
	# MARK: Forward Pass
	
	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		diff = y_hat - y
		# loss = torch.sum(torch.sqrt(diff * diff + self.eps))
		loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
		return loss
