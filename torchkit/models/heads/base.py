#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The basic functions of a head.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from torch import nn

from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import is_list_of
from torchkit.core.utils import Metrics
from torchkit.core.utils import Tensors
from torchkit.models.builder import LOSSES
from torchkit.models.builder import METRICS

logger = logging.getLogger()


# MARK: - BaseHead

class BaseHead(nn.Module, metaclass=ABCMeta):
	"""Base head.
	
	In the head, we compute the final output of the deep learning model (class
	score, bbox, ...). We also calculate losses and metrics that associate
	with the specific output.
	
	The idea behind this implementation is that we don't want to specifically
	defined the loss and metric calculate in the model class since it can be
	bothersome. We can create several heads, each with appropriate loss and
	metrics and simply attach it to the model definition.
	
	Attributes:
		cal_metrics (bool):
			Whether to calculate metrics during training. Default: `False`.
		compute_loss (nn.Module):
			Module to compute loss value.
		compute_metrics (nn.Module, Sequence[nn.Module]):
			Module to compute metrics.
	
	Args:
		loss (dict, optional):
			Config of loss. Default: `None`.
		metrics (list[dict], optional):
			Config of metrics. Default: `None`.
		cal_metrics (bool):
			Whether to calculate metrics during training. Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		loss       : Optional[dict]       = None,
		metrics    : Optional[list[dict]] = None,
		cal_metrics: bool                 = False,
		*args, **kwargs
	):
		super().__init__()
		self.cal_metrics  	 = cal_metrics
		self.compute_loss 	 = (LOSSES.build_from_dict(cfg=loss)
								if loss is not None else None)
		self.compute_metrics = (METRICS.build_from_dictlist(cfgs=metrics)
								if is_list_of(metrics, expected_type=dict)
								else [])
	
	# MARK: Forward Pass

	@abstractmethod
	def forward_xy(self, *args, **kwargs) -> ForwardXYOutput:
		"""Forward pass. Both `x` and `y` are given, hence, we compute the loss
		and metrics also.

		Args:
			kwargs (keyword arguments):
				Specific to concrete implementation.
		
		Returns:
			y_hat (Tensors):
				The final predictions tensor.
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		pass
	
	@abstractmethod
	def forward_x(self, *args, **kwargs) -> Tensors:
		"""Forward pass. During inference, only `x` is given so we compute
		`y_hat` only.

		Args:
			kwargs (keyword arguments):
				Specific to concrete implementation.
		
		Returns:
			y_hat (Tensors, Optional):
				The final predictions tensor.
		"""
		pass
	
	@abstractmethod
	def loss_metrics(self, *args, **kwargs) -> Optional[Metrics]:
		"""Calculate loss and metrics.

		Args:
			kwargs (keyword arguments):
				Specific to concrete implementation.

		Returns:
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		pass
