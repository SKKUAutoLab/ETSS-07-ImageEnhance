#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The loss function proposed in the paper "MBLLEN: Low-light Image/Video
Enhancement Using CNNs". It consists of: Structure loss, Region loss, and
Context loss.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from torchkit.core.metric import mae
from torchkit.core.metric import ssim_torch
from torchkit.models.builder import LOSSES
from torchkit.models.classifiers import VGG19

logger = logging.getLogger()

__all__ = ["MBLLENLoss", "RegionLoss", "StructureLoss", "ContextLoss"]


def region_loss(
	y_hat: torch.Tensor, y: torch.Tensor, dark_pixel_percent: float = 0.4
) -> torch.Tensor:
	"""Implementation of region loss function defined in the paper
	"MBLLEN: Low-light Image/Video Enhancement Using CNNs".
	
	Args:
		y_hat (torch.Tensor):
			The enhanced images.
		y (torch.Tensor):
			The normal-light images.
		dark_pixel_percent (float):
			Default: `0.4`.
			
	Returns:
		loss (torch.Tensor):
			The region loss tensor.
	"""
	index     = int(256 * 256 * dark_pixel_percent - 1)
	gray1     = (0.39 * y_hat[:, 0, :, :] + 0.5 * y_hat[:, 1, :, :] +
				 0.11 * y_hat[:, 2, :, :])
	gray      = torch.reshape(gray1, [-1, 256 * 256])
	gray_sort = torch.topk(-gray, k=256 * 256)[0]
	yu        = gray_sort[:, index]
	yu        = torch.unsqueeze(input=torch.unsqueeze(input=yu, dim=-1), dim=-1)
	mask      = (gray1 <= yu).type(torch.float64)
	mask1     = torch.unsqueeze(input=mask, dim=1)
	mask      = torch.cat(tensors=[mask1, mask1, mask1], dim=1)

	low_fake_clean  = torch.mul(mask, y_hat[:, :3, :, :])
	high_fake_clean = torch.mul(1 - mask, y_hat[:, :3, :, :])
	low_clean       = torch.mul(mask, y[:, : , :, :])
	high_clean      = torch.mul(1 - mask, y[:, : , :, :])
	loss            = torch.mean(torch.abs(low_fake_clean - low_clean) * 4 +
								 torch.abs(high_fake_clean - high_clean))
	
	return loss


def structure_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""Implementation of structure loss function defined in the paper
	"MBLLEN: Low-light Image/Video Enhancement Using CNNs".
	
	Args:
		y_hat (torch.Tensor):
			The enhanced images.
		y (torch.Tensor):
			The normal-light images.

	Returns:
		loss (torch.Tensor):
			The structure loss tensor.
	"""
	mae_loss  = mae(y_hat[:, :3, :, :], y)
	ssim_loss = (
		ssim_torch(y_hat=torch.unsqueeze(y_hat[:, 0, :, :], dim=1),
				   y=torch.unsqueeze(y[:, 0, :, :], dim=1), depth=1)
		+ ssim_torch(y_hat=torch.unsqueeze(y_hat[:, 1, :, :], dim=1),
					 y=torch.unsqueeze(y[:, 1, :, :], dim=1), depth=1)
		+ ssim_torch(y_hat=torch.unsqueeze(y_hat[:, 2, :, :], dim=1),
					 y=torch.unsqueeze(y[:, 2, :, :], dim=1), depth=1)
	)
	loss = mae_loss - ssim_loss
	return loss


def _range_scale(x: torch.Tensor) -> torch.Tensor:
	return x * 2.0 - 1.0


@LOSSES.register(name="context_loss")
class ContextLoss(nn.Module):
	"""Implementation of context loss function defined in the paper
	"MBLLEN: Low-light Image/Video Enhancement Using CNNs".
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "context_loss"
		self.vgg  = VGG19(out_indexes=26, pretrained=True)
		self.vgg.freeze()
		# self.context_block = FeatureBlock(module=vgg19, layer_indexes=26,
		# 									freeze=True)
	
	# MARK: Forward Pass

	# noinspection PyMethodMayBeStatic
	def forward(
		self, y_hat: torch.Tensor, y: torch.Tensor, **_
	) -> torch.Tensor:
		"""Context Loss.

		Args:
            y_hat (torch.Tensor):
				The enhanced images.
			y (torch.Tensor):
				The normal-light images.

		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		b, c, h, w     = [int(x) for x in y.shape]
		y_hat_scale    = _range_scale(y_hat)
		y_hat_features = self.vgg.forward_features(y_hat_scale)
		y_hat_features = torch.reshape(y_hat_features, shape=(-1, 16, h, w))
		
		y_scale        = _range_scale(y)
		y_features     = self.vgg.forward_features(y_scale)
		y_features     = torch.reshape(y_features, shape=(-1, 16, h, w))
		
		loss = torch.mean(
			torch.abs(y_hat_features[:, :16, :, :] - y_features[:, :16, :, :])
		)
		return loss


@LOSSES.register(name="region_loss")
class RegionLoss(nn.Module):
	"""Implementation of region loss function defined in the paper
	"MBLLEN: Low-light Image/Video Enhancement Using CNNs".
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "region_loss"
	
	# MARK: Forward Pass

	# noinspection PyMethodMayBeStatic
	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""SCR Loss.

		Args:
			y_hat (torch.Tensor):
				The enhanced images.
			y (torch.Tensor):
				The normal-light images.

		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return region_loss(y_hat=y_hat, y=y)


@LOSSES.register(name="structure_loss")
class StructureLoss(nn.Module):
	"""Implementation of structure loss function defined in the paper
	"MBLLEN: Low-light Image/Video Enhancement Using CNNs".
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "structure_loss"
	
	# MARK: Forward Pass

	# noinspection PyMethodMayBeStatic
	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""Structure Loss.

		Args:
			y_hat (torch.Tensor):
				The enhanced images.
			y (torch.Tensor):
			    The normal-light images.

		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return structure_loss(y_hat=y_hat, y=y)


@LOSSES.register(name="mbllen_loss")
class MBLLENLoss(nn.Module):
	"""Implementation of loss function defined in the paper "MBLLEN:
	Low-light Image/Video Enhancement Using CNNs".
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name 		    = "mbllen_loss"
		self.structure_loss = StructureLoss()
		self.context_loss   = ContextLoss()
		self.region_loss	= RegionLoss()

	# MARK: Forward Pass

	# noinspection PyMethodMayBeStatic
	def forward(
		self, y_hat: torch.Tensor, y: torch.Tensor, **_
	) -> torch.Tensor:
		"""The mbllen_loss = (structure_loss + (context_loss / 3.0) + 3 +
		region_loss)

		Args:
			y_hat (torch.Tensor):
				The enhanced images.
			y (torch.Tensor):
				The normal-light images.
		
		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		loss = (self.structure_loss(y_hat, y) +
				self.context_loss(y_hat, y) / 3.0 +
				3 + self.region_loss(y_hat, y))
		return loss
