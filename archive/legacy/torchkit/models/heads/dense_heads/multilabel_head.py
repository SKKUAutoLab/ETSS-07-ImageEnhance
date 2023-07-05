# ==================================================================== #
# File name: multilabel_head.py
# Author: Long H. Pham
# Date created: 08/30/2021
# The `torchkit.models.heads.dense_heads.multilabel_head` defines the
# classification head for multilabel task.
# ==================================================================== #
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import InputTensor
from torchkit.core.utils import MetricData
from torchkit.core.utils import OutputTensor
from torchkit.models.builder import HEADS
from torchkit.models.builder import LOSSES
from torchkit.models.heads.base import BaseHead

logger = logging.getLogger()


# MARK: - MultiLabelClsHead

# noinspection PyDefaultArgument
@HEADS.register(name="multilabel_cls_head")
@HEADS.register(name="multilabel_classification_head")
@HEADS.register(name="MultiLabelClsHead")
class MultiLabelClsHead(BaseHead):
	"""Classification head for multilabel task.
	
	Attributes:
		compute_loss (nn.Module):
			Module to compute loss value.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		loss    : dict           = dict(name="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0),
		name    : Optional[str]  = "multilabel_cls_head",
		init_cfg: Optional[dict] = None,
		*args, **kwargs
	):
		"""
		
		Attributes:
			loss (dict):
				Config of classification loss.
				Default: `dict(name="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0)`.
			name (str, optional):
				Name of the head. Default: `multilabel_cls_head`.
			init_cfg (dict, optional):
				The extra init config of layers. Default: `None`.
		"""
		super().__init__(
			name     = name,
			init_cfg = init_cfg,
			*args, **kwargs
		)
		
		assert isinstance(loss, dict)
		
		self.compute_loss = LOSSES.build(loss)
	
	# MARK: Forward Pass
	
	def forward_xy(self, x: InputTensor, y: InputTensor, *args, **kwargs) -> ForwardXYOutput:
		"""Classification head for multilabel task. Both `x` and `y` are given, hence, we compute the loss and metrics
		also.

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
		y     = y.type_as(y_hat)
		
		# NOTE: Calculate loss and metrics from logits
		metrics = self.loss_metrics(y_hat=y_hat, y=y)
		
		# NOTE: Calculate class-score (softmax)
		y_hat = F.softmax(y_hat, dim=1) if y_hat is not None else None

		return y_hat, metrics
	
	def forward_x(self, x: InputTensor, *args, **kwargs) -> OutputTensor:
		"""Classification head for multilabel task. During inference, only `x` is given so we compute `y_hat` only.

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
		y           = y.type_as(y_hat)
		num_samples = len(y_hat)
		metrics     = dict()
		
		# NOTE: Map difficult examples to positive ones
		_y = torch.abs(y)
		
		# NOTE: Calculate loss
		metrics["loss"] = self.compute_loss(y_hat, _y, avg_factor=num_samples)
	 
		return metrics if metrics else None
