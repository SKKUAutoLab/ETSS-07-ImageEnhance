#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SSIM evaluation metric.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .builder import METRICS

logger = logging.getLogger()


# MARK: - SSIM

def ssim_torch(
	y_hat      : torch.Tensor,
	y          : torch.Tensor,
	cs_map     : bool  = False,
	mean_metric: bool  = True,
	depth      : int   = 1,
	size       : int   = 11,
	sigma      : float = 1.5,
) -> Optional[torch.Tensor]:
	"""Calculate the SSIM (Structural Similarity Index) score between 2
	4D-/3D- channel-first- images.
    """
	y_hat = y_hat.type(torch.float64)
	y     = y.type(torch.float64)

	window = _fspecial_gauss(size=size, sigma=sigma)  # window shape [size, size]
	window = window.cuda()
	l      = depth  # depth of image (255 in case the image has a different scale)
	c1     = (0.01 * l) ** 2
	c2     = (0.03 * l) ** 2

	mu1       = F.conv2d(input=y_hat, weight=window, stride=1)
	mu2       = F.conv2d(input=y,     weight=window, stride=1)
	mu1_sq    = mu1 * mu1
	mu2_sq    = mu2 * mu2
	mu1_mu2   = mu1 * mu2
	sigma1_sq = F.conv2d(input=y_hat * y_hat, weight=window, stride=1) - mu1_sq
	sigma2_sq = F.conv2d(input=y * y,         weight=window, stride=1) - mu2_sq
	sigma12   = F.conv2d(input=y_hat * y,     weight=window, stride=1) - mu1_mu2

	if cs_map:
		score = (
			((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
			((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)),
			(2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
		)
	else:
		score = (
			((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
			/ ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
		)

	if mean_metric:
		score = torch.mean(score)
	score = score.detach()
	return score


def multiscale_ssim_torch(
	y_hat      : torch.Tensor,
	y          : torch.Tensor,
	mean_metric: bool = True,
	level      : int  = 5,
) -> torch.Tensor:
	"""
	
	Args:
		y_hat (torch.Tensor):
			4D-/3D- channel-first- images.
		y (torch.Tensor):
			4D-/3D- channel-first- images.
		mean_metric (bool):
			Average the ssim scores.
		level (int):
			 Default: `5`.

	Returns:
		score (torch.Tensor):
			The SSIM score.
	"""
	y_hat  = y_hat.detach().cpu()
	y      = y.detach().cpu()
	weight = torch.Tensor(
		[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float64)
	mssim  = []
	mcs    = []

	for l in range(level):
		ssim_map, cs_map = ssim_torch(
			y_hat=y_hat, y=y, cs_map=True, mean_metric=False
		)
		mssim.append(torch.mean(ssim_map))
		mcs.append(torch.mean(cs_map))
		filtered_y_hat = F.avg_pool2d(
			input=y_hat, kernel_size=(2, 2), stride=(2, 2), padding=1,
			padding_mode="replicate"
		)
		filtered_y = F.avg_pool2d(
			input=y, kernel_size=(2, 2), stride=(2, 2), padding=1,
			padding_mode="replicate"
		)
		y_hat = filtered_y_hat
		y     = filtered_y

	# List to tensor of dim D+1
	mssim = torch.stack(mssim, dim=0)
	mcs   = torch.stack(mcs,   dim=0)

	score = (
		torch.prod(mcs[0:level - 1] ** weight[0:level - 1])
		* (mssim[level - 1] ** weight[level - 1])
	)

	if mean_metric:
		score = torch.mean(score)
	score = score.detach()
	return score


# noinspection PyTypeChecker
def _fspecial_gauss(size: int, sigma: float) -> torch.Tensor:
	"""Function to mimic the `fspecial` gaussian MATLAB function.

	Args:
		size (int):
			The size of gaussian's window. Default: `11`.
		sigma (float):
			The sigma value of gaussian's window. Default: `1.5`.
	"""
	x_data, y_data = np.mgrid[
						 -size // 2 + 1: size // 2 + 1,
						 -size // 2 + 1: size // 2 + 1
					 ]
	x_data = np.expand_dims(x_data, axis=0)
	x_data = np.expand_dims(x_data, axis=0)
	y_data = np.expand_dims(y_data, axis=0)
	y_data = np.expand_dims(y_data, axis=0)
	x      = torch.from_numpy(x_data)
	y      = torch.from_numpy(y_data)
	x      = x.type(torch.float64)
	y      = y.type(torch.float64)
	z      = -((x ** 2 + y ** 2) / (2.0 * sigma ** 2))
	g      = torch.exp(z)
	return g / torch.sum(g)


@METRICS.register(name="ssim")
class SSIM(nn.Module):
	"""Calculate the SSIM (Structural Similarity Index).
	
	Attributes:
		cs_map (bool):
			Default: `False`.
		mean_metric (bool):
			Average the ssim scores.
		depth (int):
			Depth of image. Default: `1` (`255` in case the image has a
			different scale).
		size (int):
			The size of gaussian's window.
		sigma (float):
			The sigma value of gaussian's window.
		name (str):
			Name of the loss.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		cs_map     : bool          = False,
		mean_metric: bool          = True,
		depth      : int           = 1,
		size       : int           = 11,
		sigma      : float         = 1.5,
		name       : Optional[str] = "ssim",
	):
		super().__init__()
		self.name 	     = "ssim"
		self.cs_map      = cs_map
		self.mean_metric = mean_metric
		self.depth       = depth
		self.size        = size
		self.sigma       = sigma

	# MARK: Forward Pass

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return ssim_torch(
			y_hat=y_hat, y=y, cs_map=self.cs_map, mean_metric=self.mean_metric,
			depth=self.depth, size=self.size, sigma=self.sigma,
		)


@METRICS.register(name="multiscale_ssim")
@METRICS.register(name="MultiscaleSSIM")
class MultiscaleSSIM(nn.Module):
	"""Calculate the Multiscale SSIM (Structural Similarity Index).

	Attributes:
		mean_metric (bool):
			Average the ssim scores. Default: `True`.
		level (int):
			Default: `5`.
		name (str):
			Name of the loss.
	"""

	# MARK: Magic Functions

	def __init__(self, mean_metric: bool = True, level: int = 5):
		super().__init__()
		self.name 		 = "multiscale_ssim"
		self.mean_metric = mean_metric
		self.level       = level

	# MARK: Forward Pass

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return multiscale_ssim_torch(
			y_hat=y_hat, y=y, mean_metric=self.mean_metric, level=self.level,
		)
