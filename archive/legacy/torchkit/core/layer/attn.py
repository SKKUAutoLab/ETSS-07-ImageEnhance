#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention Layers.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchkit.core.utils import Size2T
from torchkit.core.utils import to_2tuple

from .builder import ATTN_LAYERS

logger = logging.getLogger()


# MARK: - Register

ATTN_LAYERS.register(name="identity", module=nn.Identity)


# MARK: - ChannelAttentionLayer

@ATTN_LAYERS.register(name="channel_attention_layer")
class ChannelAttentionLayer(nn.Module):
	"""Channel Attention Layer.
	
	Attributes:
		channels (int):
			Number of input and output channels.
		reduction (int):
			Reduction factor. Default: `16`.
		bias (bool):
			Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, channels: int, reduction: int = 16, bias: bool = False):
		super().__init__()
		# Global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# Feature channel downscale and upscale --> channel weight
		self.conv_du  = nn.Sequential(
			nn.Conv2d(channels, channels // reduction, kernel_size=(1, 1),
					  padding=0, bias=bias),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction, channels, kernel_size=(1, 1),
					  padding=0, bias=bias),
			nn.Sigmoid()
		)
	
	# MARK: Forward Pass
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y     = self.avg_pool(x)
		y     = self.conv_du(y)
		y_hat = x * y
		return y_hat


CAL = ChannelAttentionLayer
ATTN_LAYERS.register(name="cal", module=CAL)


# MARK: - ChannelAttentionBlock

@ATTN_LAYERS.register(name="channel_attention_block")
class ChannelAttentionBlock(nn.Module):
	"""Channel Attention Block.
	
	Attributes:
		channels (int):
			Number of input and output channels.
		kernel_size (Size2T):
			The kernel size of the convolution layer.
		reduction (int):
			Reduction factor.
		bias (bool):
			Default: `False`.
		act (nn.Module):
			The activation layer.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels   : int,
		kernel_size: Size2T,
		reduction  : int,
		bias       : bool,
		act        : nn.Module,
	):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding     = kernel_size[0] // 2
		stride		= (1, 1)
		self.ca     = CAL(channels, reduction, bias)
		self.body   = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size, stride, padding,
					  bias=bias),
			act,
			nn.Conv2d(channels, channels, kernel_size, stride, padding,
					  bias=bias),
		)
		
	# MARK: Forward Pass
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y     = self.body(x)
		y     = self.ca(y)
		y_hat = y + x
		return y_hat


CAB = ChannelAttentionBlock
ATTN_LAYERS.register(name="cab")


# MARK: - SupervisedAttentionModule

@ATTN_LAYERS.register(name="supervised_attention_module")
class SupervisedAttentionModule(nn.Module):
	"""Supervised Attention Module.
	
	Args:
		channels (int):
			Number of input channels.
		kernel_size (Size2T):
			The kernel size of the convolution layer.
		bias (bool):
			Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, channels: int, kernel_size: Size2T, bias: bool):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding     = kernel_size[0] // 2
		stride		= (1, 1)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride,
							   padding, bias=bias)
		self.conv2 = nn.Conv2d(channels, 3,        kernel_size, stride,
							   padding, bias=bias)
		self.conv3 = nn.Conv2d(3,        channels, kernel_size, stride,
							   padding, bias=bias)
		
	# MARK: Forward Pass
	
	def forward(
		self, fx: torch.Tensor, x: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Run forward pass.

		Args:
			fx (torch.Tensor):
				The output from previous steps.
			x (torch.Tensor):
				The original input images.
			
		Returns:
			y_hat (torch.Tensor):
				The output tensor.
			img (torch.Tensor):
				The output image tensor.
		"""
		x1    = self.conv1(fx)
		img   = self.conv2(fx) + x
		x2    = torch.sigmoid(self.conv3(img))
		x1    = x1 * x2
		x1    = x1 + fx
		y_hat = x1
		return y_hat, img


SAM = SupervisedAttentionModule
ATTN_LAYERS.register(name="sam")
