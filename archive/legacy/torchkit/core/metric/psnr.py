#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSNR evaluation metric
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import nn

from .builder import METRICS

logger = logging.getLogger()


# MARK: - PSNR

def psnr_numpy(y_hat: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
	""""Calculate peak signal-to-noise ratio score between 2 4D-/3D-
	channel-first- images.
    """
	imdff = np.float64(y_hat) - np.float64(y)
	rmse  = np.sqrt(np.mean(imdff ** 2))
	if rmse == 0:
		return None
	score = 20 * np.log10(255.0 / rmse)
	return score


def psnr_torch(y_hat: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
	""""Calculate peak signal-to-noise ratio score between 2 4D-/3D-
	channel-first- images.
    """
	imdff = torch.clamp(y_hat, 0, 1) - torch.clamp(y, 0, 1)
	rmse  = torch.sqrt(torch.mean(imdff ** 2))
	if rmse == 0:
		return None
	score = 20 * torch.log10(1.0 / rmse)
	score = score.detach()
	return score


def psnr(
	y_hat: Union[torch.Tensor, np.ndarray],
	y    : Union[torch.Tensor, np.ndarray],
) -> Optional[Union[torch.Tensor, np.ndarray]]:
	""""Calculate peak signal-to-noise ratio score between 2 4D-/3D-
	channel-first- images.
    """
	if isinstance(y_hat, torch.Tensor) and isinstance(y, torch.Tensor):
		score = psnr_torch(y_hat=y_hat, y=y)
	elif isinstance(y_hat, np.ndarray) and isinstance(y, np.ndarray):
		score = psnr_numpy(y_hat=y_hat, y=y)
	else:
		raise TypeError(
			f"`y_hat` and `y` should both be `torch.Tensor` or "
			f"`np.ndarray`, but got {type(y_hat)} and {type(y)}."
		)
	return score


# noinspection PyMethodMayBeStatic
@METRICS.register(name="psnr")
class PSNR(nn.Module):
	"""Calculate peak signal-to-noise ratio.

	Attributes:
		name (str):
			Name of the metric.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "psnr"
	
	# MARK: Forward Pass
	
	def forward(
		self,
		y_hat: Union[torch.Tensor, np.ndarray],
		y    : Union[torch.Tensor, np.ndarray],
	) -> Optional[Union[torch.Tensor, np.ndarray]]:
		return psnr(y_hat=y_hat, y=y)
