#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Focal Loss.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .builder import LOSSES
from .utils import weight_reduce_loss

logger = logging.getLogger()


# MARK: - FocalLoss

def sigmoid_focal_loss(
	y_hat     : torch.Tensor,
	y         : torch.Tensor,
	weight    : Optional[torch.Tensor] = None,
	gamma     : float                  = 2.0,
	alpha     : float                  = 0.25,
	reduction : str                    = "mean",
	avg_factor: Optional[int]          = None,
) -> torch.Tensor:
	"""Sigmoid focal loss.
	
	Args:
		y_hat (torch.Tensor):
			The prediction with shape (N, *).
		y (torch.Tensor):
			The ground truth label of the prediction with shape (N, *).
		weight (torch.Tensor, optional):
			Sample-wise loss weight with shape (N, ).
		gamma (float):
			The gamma for calculating the modulating factor.
		alpha (float):
			A balanced form for Focal Loss.
		reduction (str):
			The method used to reduce the loss. One of: [`none`, `mean`, `sum`].
			If reduction is `none`, loss is same shape as pred and label.
		avg_factor (int, optional):
			Average factor that is used to average the loss.
			
	Returns:
		loss (torch.Tensor):
			The calculated loss.
	"""
	assert y_hat.shape == y.shape, \
		"`pred` and `target` should be in the same shape."

	pred_sigmoid = y_hat.sigmoid()
	y            = y.type_as(y_hat)
	pt           = (1 - pred_sigmoid) * y + pred_sigmoid * (1 - y)
	focal_weight = (alpha * y + (1 - alpha) * (1 - y)) * pt.pow(gamma)

	loss = (
		F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
		* focal_weight
	)
	if weight is not None:
		assert weight.dim() == 1
		weight = weight.float()
		if y_hat.dim() > 1:
			weight = weight.reshape(-1, 1)

	loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
	return loss


@LOSSES.register(name="focal_loss")
class FocalLoss(nn.Module):
	"""Focal loss.
	
	Attributes:
		gamma (float):
			Focusing parameter in focal loss.
		alpha (float):
			The parameter in balanced form of focal loss.
		reduction (str):
			The method used to reduce the loss. One of: [`none`, `mean`, `sum`].
			If reduction is `none`, loss is same shape as pred and label.
		loss_weight (float):
			Weight of the loss.
		name (str):
			Name of the loss.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		gamma      : float = 2.0,
		alpha      : float = 0.25,
		reduction  : str   = "mean",
		loss_weight: float = 1.0,
	):
		super().__init__()
		self.name 		 = "focal_loss"
		self.gamma       = gamma
		self.alpha       = alpha
		self.reduction   = reduction
		self.loss_weight = loss_weight
	
	# MARK: Forward Pass
	
	def forward(
		self,
		y_hat             : torch.Tensor,
		y                 : torch.Tensor,
		weight            : Optional[torch.Tensor] = None,
		avg_factor        : Optional[int]          = None,
		reduction_override: Optional[str]          = None,
	) -> torch.Tensor:
		assert reduction_override in (None, "none", "mean", "sum")
		reduction = (
			reduction_override if reduction_override else self.reduction
		)
		loss_cls  = (
			self.loss_weight *
			sigmoid_focal_loss(
				y_hat=y_hat, y=y, weight=weight, gamma=self.gamma,
				alpha=self.alpha, reduction=reduction, avg_factor=avg_factor
			)
		)
		return loss_cls
