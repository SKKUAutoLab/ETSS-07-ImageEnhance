#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Resolution Block used in paper: `Multi-Stage Progressive Image Restoration`.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from torchkit.core.utils import Size2T
from torchkit.core.utils import to_2tuple
from .attn import CAB

logger = logging.getLogger()


# MARK: - Original Resolution Block

# noinspection PyTypeChecker
class OriginalResolutionBlock(nn.Module):
	"""Original Resolution Block.
	
	Args:
		channels (int):
			Number of input and output channels.
		kernel_size (Size2T):
			The kernel size of the convolution layer.
		reduction (int):
			Reduction factor. Default: `16`.
		bias (bool):
			Default: `False`.
		act (nn.Module):
			The activation function.
		num_cab (int):
			Number of CAB modules used.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels   : int,
		kernel_size: Size2T,
		reduction  : int,
		bias       : bool,
		act        : nn.Module,
		num_cab    : int,
	):
		
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding 	= kernel_size[0] // 2
		body        = [CAB(channels, kernel_size, reduction, bias, act)
					   for _ in range(num_cab)]
		body.append(
			nn.Conv2d(channels, channels, kernel_size, stride=(1, 1),
					  padding=padding, bias=bias)
		)
		self.body = nn.Sequential(*body)
	
	# MARK: Forward Pass
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y_hat  = self.body(x)
		y_hat += x
		return y_hat


ORB = OriginalResolutionBlock
