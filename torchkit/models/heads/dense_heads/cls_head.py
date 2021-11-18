#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classification head.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch.nn.functional as F

from torchkit.core.metric import Accuracy
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import Tensors
from torchkit.core.utils import Metrics
from torchkit.models.builder import HEADS
from torchkit.models.builder import LOSSES
from torchkit.models.heads.base import BaseHead

logger = logging.getLogger()


# MARK: - ClsHead

# noinspection PyDefaultArgument
@HEADS.register(name="cls_head")
class ClsHead(BaseHead):
	"""Classification head.
	
	Attributes:
		top_k (int, tuple[int]):
			`top-k` accuracy. Default: `(1, )`.
		cal_acc (bool):
			Whether to calculate accuracy during training. If you use
			Mixup/CutMix or something like that during training, it is not
			reasonable to calculate accuracy. Default: `False`.
		compute_loss (nn.Module):
			Module to compute loss value.
		compute_accuracy (nn.Module):
			Module to compute accuracy.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		top_k   : Union[int, tuple[int]] = (1, ),
		cal_acc : bool = False,
		loss    : dict = dict(name="CrossEntropyLoss", loss_weight=1.0),
		*args, **kwargs
	):
		super().__init__(
			name     = name,
			init_cfg = init_cfg,
			*args, **kwargs
		)
	 
		assert isinstance(loss, dict)
		assert isinstance(top_k, (int, tuple))
		if isinstance(top_k, int):
			top_k = (top_k, )
		for _top_k in top_k:
			assert _top_k > 0, "`top-k` should be larger than 0."
	
		self.top_k            = top_k
		self.cal_acc          = cal_acc
		self.compute_loss     = LOSSES.build(cfg=loss)
		self.compute_accuracy = Accuracy(topk=self.top_k)
		
	# MARK: Forward Pass
	
	def forward_xy(self, x: InputTensor, y: InputTensor, *args, **kwargs) -> ForwardXYOutput:
		"""Classification head. Both `x` and `y` are given, hence, we compute the loss and metrics also.

		Args:
			x (InputTensor):
				`x` contains either the input data or the predictions from previous step.
			y (InputTensor):
				`y` contains the ground truth.

		Returns:
			(ForwardXYOutput):
				y_hat (OutputTensor):
					The final predictions tensor.
				metrics (MetricData):
					- A dictionary with the first key must be the `loss`.
					- `None`, training will skip to the next batch.
		"""
		if isinstance(x, tuple):
			x = x[-1]
		y_hat = x
		
		# NOTE: Calculate loss and metrics from logits
		metrics = self.loss_metrics(y_hat=y_hat, y=y)
		
		# NOTE: Calculate class-score (softmax)
		y_hat   = F.softmax(y_hat, dim=1) if y_hat is not None else None
		
		return y_hat, metrics
	
	def forward_x(self, x: InputTensor, *args, **kwargs) -> OutputTensor:
		"""Classification head. During inference, only `x` is given so we compute `y_hat` only.

		Args:
			x (InputTensor):
				`x` contains either the input data or the predictions from previous step.

		Returns:
			y_hat (OutputTensor):
				The final prediction.
		"""
		if isinstance(x, tuple):
			x = x[-1]
		if isinstance(x, list):
			x = sum(x) / float(len(x))
		y_hat = x
		
		# NOTE: Calculate class-score (softmax)
		y_hat = F.softmax(y_hat, dim=1) if y_hat is not None else None
	
		return y_hat
	
	def loss_metrics(self, y_hat: InputTensor, y: InputTensor, *args, **kwargs) -> Optional[MetricData]:
		"""Calculate loss and metrics.

		Args:
			y_hat (InputTensor):
				`y_hat` contains the predictions of the model.
			y (InputTensor):
				`y` contains the ground truth.

		Returns:
			metrics (MetricData, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		num_samples = len(y_hat)
		metrics     = dict()
		
		# NOTE: Calculate loss
		metrics["loss"] = self.compute_loss(y_hat, y, avg_factor=num_samples)
		
		# NOTE: Calculate metrics
		if self.cal_acc:
			acc = self.compute_accuracy(y_hat, y)
			assert len(acc) == len(self.top_k)
			metrics["accuracy"] = {f"top_{k}": a for k, a in zip(self.top_k, acc)}
		
		return metrics if metrics else None
