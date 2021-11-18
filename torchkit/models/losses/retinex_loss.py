#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The decomposition loss and enhancement loss according to RetinexNet
paper (https://arxiv.org/abs/1808.04560).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchkit.core.image import avg_gradient
from torchkit.core.image import gradient
from torchkit.models.builder import LOSSES

logger = logging.getLogger()


def decom_loss(
	x     : torch.Tensor,
	y     : torch.Tensor,
	r_low : torch.Tensor,
	r_high: torch.Tensor,
	i_low : torch.Tensor,
	i_high: torch.Tensor,
	**_
) -> torch.Tensor:
	"""Calculate the decomposition loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).

	Args:
		x (torch.Tensor):
			The low-light images.
		y (torch.Tensor):
			The normal-light images.
		r_low (torch.Tensor):
			The reflectance map extracted from low-light images.
		r_high (torch.Tensor):
			The reflectance map extracted from normal-light images.
		i_low (torch.Tensor):
			The illumination map extracted from low-light images as a single
			channel.
		i_high (torch.Tensor):
			The illumination map extracted from normal-light images as a single
			channel.
			
	Returns:
		loss (torch.Tensor):
			The loss tensor.
	"""
	i_low_3                = torch.cat(tensors=(i_low,  i_low,  i_low),  dim=1)
	i_high_3               = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
	recon_loss_low         = F.l1_loss(r_low * i_low_3, x)
	recon_loss_high        = F.l1_loss(r_high * i_high_3, y)
	recon_loss_mutual_low  = F.l1_loss(r_high * i_low_3, x)
	recon_loss_mutual_high = F.l1_loss(r_low * i_high_3, y)
	equal_r_loss           = F.l1_loss(r_low, r_high)
	i_smooth_loss_low      = _smooth(i_low, r_low)
	i_smooth_loss_high     = _smooth(i_high, r_high)
	
	loss = (recon_loss_low +
		   recon_loss_high +
		   0.001 * recon_loss_mutual_low +
		   0.001 * recon_loss_mutual_high +
		   0.1   * i_smooth_loss_low +
		   0.1   * i_smooth_loss_high +
		   0.01  * equal_r_loss)
	return loss


def enhance_loss(
	y        : torch.Tensor,
	r_low    : torch.Tensor,
	i_delta  : torch.Tensor,
	i_delta_3: Optional[torch.Tensor] = None,
	y_hat    : Optional[torch.Tensor] = None,
	**_
) -> torch.Tensor:
	"""Calculate the enhancement loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).

	Args:
		y (torch.Tensor):
			The normal-light images.
		r_low (torch.Tensor):
			The reflectance map extracted from low-light images.
		i_delta (torch.Tensor):
			The enhanced illumination map produced from low-light images as a
			single-channel.
		i_delta_3 (torch.Tensor, optional):
			The enhanced illumination map produced from low-light images as a
			3-channels. Default: `None`.
		y_hat (torch.Tensor, optional):
			The enhanced low-light images. Default: `None`.
			
	Returns:
		loss (torch.Tensor):
			The enhance loss tensor.
	"""
	i_delta_3    		= (torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
						   if i_delta_3 is None else i_delta_3)
	y_hat  		 		= (r_low * i_delta_3) if y_hat is None else y_hat
	relight_loss 		= F.l1_loss(y_hat, y)
	i_smooth_loss_delta = _smooth(i_delta, r_low)
	loss 				= relight_loss + 3 * i_smooth_loss_delta
	return loss


def retinex_loss(
	x      : torch.Tensor,
	y      : torch.Tensor,
	r_low  : torch.Tensor,
	r_high : torch.Tensor,
	i_low  : torch.Tensor,
	i_high : torch.Tensor,
	i_delta: torch.Tensor,
	**_
) -> torch.Tensor:
	"""Calculate the combined decom loss and enhance loss.

	Args:
		x (torch.Tensor):
			The low-light images.
		y (torch.Tensor):
			The normal-light images.
		r_low (torch.Tensor):
			The reflectance map extracted from low-light images.
		r_high (torch.Tensor):
			The reflectance map extracted from normal-light images.
		i_low (torch.Tensor):
			The illumination map extracted from low-light images as a single
			channel.
		i_high (torch.Tensor):
			The illumination map extracted from normal-light images as a single
			channel.
		i_delta (torch.Tensor):
			The enhanced illumination map produced from low-light images as a
			single-channel.

	Returns:
		loss (torch.Tensor):
			The combined loss tensor.
	"""
	loss1 = decom_loss(
		x      = x,
		y      = y,
		r_low  = r_low,
		r_high = r_high,
		i_low  = i_low,
		i_high = i_high,
	)
	loss2 = enhance_loss(
		y       = y,
		r_low   = r_low,
		i_delta = i_delta,
	)
	loss = loss1 + loss2
	return loss


def _smooth(i: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
	"""Get the smooth reconstructed image from the given illumination map and reflectance map.
	
	Args:
		i (torch.Tensor):
			The illumination map.
		r (torch.Tensor):
			The reflectance map.

	Returns:
		grad (torch.Tensor):
			The smoothed reconstructed image.
	"""
	r    = (0.299 * r[:, 0, :, :]) + (0.587 * r[:, 1, :, :]) + (0.114 * r[:, 2, :, :])
	r    = torch.unsqueeze(input=r, dim=1)
	grad = gradient(input=i, direction="x") * torch.exp(-10 * avg_gradient(input=r, direction="x")) + \
	       gradient(input=i, direction="y") * torch.exp(-10 * avg_gradient(input=r, direction="y"))
	return torch.mean(input=grad)


# noinspection PyMethodMayBeStatic
@LOSSES.register(name="decom_loss")
class DecomLoss(nn.Module):
	"""Calculate the decomposition loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "decom_loss"
	
	# MARK: Forward Pass
	
	def forward(
		self,
		x     : torch.Tensor,
		y     : torch.Tensor,
		r_low : torch.Tensor,
		r_high: torch.Tensor,
		i_low : torch.Tensor,
		i_high: torch.Tensor,
		**_
	) -> torch.Tensor:
		"""Forward pass.
		
		Args:
			x (torch.Tensor):
				The low-light images.
			y (torch.Tensor):
				The normal-light images.
			r_low (torch.Tensor):
				The reflectance map extracted from low-light images.
			r_high (torch.Tensor):
				The reflectance map extracted from normal-light images.
			i_low (torch.Tensor):
				The illumination map extracted from low-light images as a
				single channel.
			i_high (torch.Tensor):
				The illumination map extracted from normal-light images as a
				single channel.
				
		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return decom_loss(
			x      = x,
			y      = y,
			r_low  = r_low,
			r_high = r_high,
			i_low  = i_low,
			i_high = i_high,
		)
	
	
# noinspection PyMethodMayBeStatic
@LOSSES.register(name="enhance_loss")
class EnhanceLoss(nn.Module):
	"""Calculate the enhancement loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "enhance_loss"
	
	# MARK: Forward Pass
	
	def forward(
		self,
		y        : torch.Tensor,
		r_low    : torch.Tensor,
		i_delta  : torch.Tensor,
		i_delta_3: Optional[torch.Tensor] = None,
		y_hat    : Optional[torch.Tensor] = None,
		**_
	) -> torch.Tensor:
		"""Run forward pass.

		Args:
			y (torch.Tensor):
				The normal-light images.
			r_low (torch.Tensor):
				The reflectance map extracted from low-light images.
			i_delta (torch.Tensor):
				The enhanced illumination map produced from low-light images
				as a single-channel.
			i_delta_3 (torch.Tensor, optional):
				The enhanced illumination map produced from low-light images
				as a 3-channels. Default: `None`.
			y_hat (torch.Tensor, optional):
				The enhanced low-light images. Default: `None`.
				
		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return enhance_loss(
			y         = y,
			r_low     = r_low,
			i_delta   = i_delta,
			i_delta_3 = i_delta_3,
			y_hat     = y_hat,
		)


# noinspection PyMethodMayBeStatic
@LOSSES.register(name="retinex_loss")
class RetinexLoss(nn.Module):
	"""Calculate the combined decomposition loss and enhancement loss."""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "retinex_loss"
		
	# MARK: Forward Pass
	
	def forward(
		self,
		x      : torch.Tensor,
		y      : torch.Tensor,
		r_low  : torch.Tensor,
		r_high : torch.Tensor,
		i_low  : torch.Tensor,
		i_high : torch.Tensor,
		i_delta: torch.Tensor,
		**_
	) -> torch.Tensor:
		"""Run forward pass.

		Args:
			x (torch.Tensor):
				The low-light images.
			y (torch.Tensor):
				The normal-light images.
			r_low (torch.Tensor):
				The reflectance map extracted from low-light images.
			r_high (torch.Tensor):
				The reflectance map extracted from normal-light images.
			i_low (torch.Tensor):
				The illumination map extracted from low-light images as a
				single channel.
			i_high (torch.Tensor):
				The illumination map extracted from normal-light images as a
				single channel.
			i_delta (torch.Tensor):
				The enhanced illumination map produced from low-light images
				as a single-channel.
	
		Returns:
			loss (torch.Tensor):
				The loss tensor.
		"""
		return retinex_loss(
			x       = x,
			y       = y,
			r_low   = r_low,
			r_high  = r_high,
			i_low   = i_low,
			i_high  = i_high,
			i_delta = i_delta,
		)
