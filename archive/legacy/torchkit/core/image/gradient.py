#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Gradient Operations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# MARK: - Functional API

def gradient(input: torch.Tensor, direction: str) -> torch.Tensor:
	"""Calculate the gradient in the image with the desired direction.

	Args:
		input (torch.Tensor):
			The input image tensor.
		direction (str):
			The direction to calculate the gradient. Can be ["x", "y"].

	Returns:
		grad (torch.Tensor)
	"""
	smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
	smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
	
	if direction == "x":
		kernel = smooth_kernel_x
	elif direction == "y":
		kernel = smooth_kernel_y
	
	kernel = kernel.cuda()
	grad   = torch.abs(
		F.conv2d(input=input, weight=kernel, stride=1, padding=1)
	)
	return grad


def avg_gradient(input: torch.Tensor, direction: str) -> torch.Tensor:
	"""Calculate the average gradient in the image with the desired direction.

	Args:
		input (torch.Tensor):
			The input image tensor.
		direction (str):
			The direction to calculate the gradient. Can be ["x", "y"].

	Returns:
		avg_gradient (torch.Tensor):
			The average gradient.
	"""
	return F.avg_pool2d(gradient(input=input, direction=direction),
						kernel_size=3, stride=1, padding=1)
