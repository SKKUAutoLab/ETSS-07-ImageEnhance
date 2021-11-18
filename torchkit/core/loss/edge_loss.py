#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Edge Loss.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from .builder import LOSSES
from .charbonnier_loss import CharbonnierLoss

logger = logging.getLogger()


# MARK: - EdgeLoss

@LOSSES.register(name="edge_loss")
class EdgeLoss(nn.Module):
	"""Edge Loss.

	Attributes:
		name (str):
			Name of the loss.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
		self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
		if torch.cuda.is_available():
			self.kernel = self.kernel.cuda()

		self.loss = CharbonnierLoss()
		self.name = "edge_loss"
	
	# MARK: Forward Pass
	
	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		loss = self.loss(self.laplacian_kernel(y_hat), self.laplacian_kernel(y))
		return loss
	
	def conv_gauss(self, img):
		n_channels, _, kw, kh = self.kernel.shape
		img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
		return F.conv2d(img, self.kernel, groups=n_channels)

	def laplacian_kernel(self, current):
		filtered    = self.conv_gauss(current)     # filter
		down        = filtered[:, :, ::2, ::2]     # downsample
		new_filter  = torch.zeros_like(filtered)
		new_filter[:, :, ::2, ::2] = down * 4      # upsample
		filtered    = self.conv_gauss(new_filter)  # filter
		diff        = current - filtered
		return diff
