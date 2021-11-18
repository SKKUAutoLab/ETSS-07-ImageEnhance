#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The loss function proposed in the paper "Multi-Stage Progressive Image
Restoration".
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from torchkit.core.loss import CharbonnierLoss
from torchkit.core.loss import EdgeLoss
from torchkit.core.utils import Tensors
from torchkit.models.builder import LOSSES

logger = logging.getLogger()


def mpr_loss(y_hat: Tensors, y: torch.Tensor) -> torch.Tensor:
	"""MPR Loss.
	
	Args:
		y_hat (Tensors):
			The sequence of enhanced images combined from 3 different
				branches.
		y (torch.Tensor):
			The normal-light images.
	
	Returns:
		loss (torch.Tensor):
			The loss tensor.
	"""
	charbonnier_loss_ = sum([CharbonnierLoss()(y_hat_, y) for y_hat_ in y_hat])
	edge_loss_        = sum([EdgeLoss()(y_hat_, y)        for y_hat_ in y_hat])
	loss              = charbonnier_loss_ + 0.05 * edge_loss_
	return loss


@LOSSES.register(name="mpr_loss")
class MPRLoss(nn.Module):
	"""Implementation of the loss function proposed in the paper
	"Multi-Stage Progressive Image Restoration".
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "mpr_loss"
	
	# MARK: Forward Pass

	# noinspection PyMethodMayBeStatic
	def forward(self, y_hat: Tensors, y: torch.Tensor, **_) -> torch.Tensor:
		"""MPR Loss.

		Args:
			y_hat (Tensors):
				The sequence of enhanced images combined from 3 different
				branches.
			y (torch.Tensor):
				The normal-light images.

		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return mpr_loss(y_hat=y_hat, y=y)
