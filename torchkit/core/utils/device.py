#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from .style_print import prints

logger = logging.getLogger()


# MARK: - Query

def select_device(
	model_name: str           = "",
	device    : Optional[str] = "",
	batch_size: Optional[int] = None
) -> torch.device:
	"""Select the device to runners the model.
	
	Args:
		model_name (str):
			Name of the model.
		device (str, optional):
			Name of device for running.
		batch_size (int, optional):
			Number of samples in one forward & backward pass.

	Returns:
		device (torch.device):
			GPUs or CPU.
	"""
	if device is None:
		return torch.device("cpu")

	# device = 'cpu' or '0' or '0,1,2,3'
	s   = f"{model_name}"  # string
	
	if isinstance(device, str) and device.lower() == "cpu":
		cpu = True
	else:
		cpu = False
	
	if cpu:
		# Force torch.cuda.is_available() = False
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	elif device:
		# Non-cpu device requested
		os.environ["CUDA_VISIBLE_DEVICES"] = device
		# Check availability
		assert torch.cuda.is_available(), \
			f"CUDA unavailable, invalid device {device} requested"
	
	cuda = not cpu and torch.cuda.is_available()
	
	if cuda:
		n = torch.cuda.device_count()

		# Check that batch_size is compatible with device_count
		if n > 1 and batch_size:
			assert batch_size % n == 0,\
				f"batch-size {batch_size} not multiple of GPU count {n}"
		space = " " * len(s)
		
		for i, d in enumerate(device.split(",") if device else range(n)):
			p = torch.cuda.get_device_properties(i)
			s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
	else:
		s += 'CPU\n'
	
	prints(s)  # skip a line
	return torch.device("cuda:0" if cuda else "cpu")
