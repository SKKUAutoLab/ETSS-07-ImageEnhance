# ==================================================================== #
# File name: multilabel_linear_head.py
# Author: Long H. Pham
# Date created: 08/30/2021
# The `torchkit.models.heads.dense_heads.multilabel_linear_head`
# defines the linear classification head for multilabel task.
# ==================================================================== #
from __future__ import annotations

import logging
from typing import Optional

import torch.nn.functional as F
from torch import nn

from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import InputTensor
from torchkit.core.utils import OutputTensor
from torchkit.models.builder import HEADS
from .multilabel_head import MultiLabelClsHead

logger = logging.getLogger()


# MARK: - MultiLabelLinearClsHead

# noinspection PyDefaultArgument
@HEADS.register(name="multilabel_linear_cls_head")
@HEADS.register(name="multilabel_linear_classification_head")
@HEADS.register(name="MultiLabelLinearClsHead")
class MultiLabelLinearClsHead(MultiLabelClsHead):
	"""Linear classification head for multilabel task.

	Attributes:
		in_channels (int):
            Number of channels in the input feature map.
		num_classes (int):
			Number of categories.
        fc (nn.Module):
            The fully-connected layer.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		in_channels: int,
		num_classes: int,
		loss       : dict           = dict(name="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0),
		name       : Optional[str]  = "multilabel_linear_cls_head",
		init_cfg   : Optional[dict] = dict(name="Normal", layer="Linear", std=0.01),
		*args, **kwargs
	):
		"""
		
		Args:
			in_channels (int):
	            Number of channels in the input feature map.
			num_classes (int):
				Number of categories.
			loss (dict):
				Config of classification loss.
				Default: `dict(name="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0)`.
			name (str, optional):
				Name of the head. Default: `multilabel_linear_cls_head`.
			init_cfg (dict, optional):
				The extra init config of layers. Default: `dict(name="Normal", layer="Linear", std=0.01)`.
		"""
		super().__init__(
			loss     = loss,
			name     = name,
			init_cfg = init_cfg,
			*args, **kwargs
		)
		
		if num_classes <= 0:
			raise ValueError(f"num_classes={num_classes} must be a positive integer.")
		
		self.in_channels = in_channels
		self.num_classes = num_classes
		self.fc          = nn.Linear(in_features=self.in_channels, out_features=self.num_classes)
	
	# MARK: Forward Pass
	
	def forward_xy(self, x: InputTensor, y: InputTensor, *args, **kwargs) -> ForwardXYOutput:
		"""Linear classification head for multilabel task. Both `x` and `y` are given, hence, we compute the loss and
		metrics also.

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
		y     = y.type_as(x)
		y_hat = self.fc(x)
		
		# NOTE: Calculate loss and metrics from logits
		metrics = self.loss_metrics(y_hat=y_hat, y=y)
		
		# NOTE: Calculate class-score (softmax)
		y_hat = F.sigmoid(y_hat) if y_hat is not None else None

		return y_hat, metrics
	
	def forward_x(self, x: InputTensor, *args, **kwargs) -> OutputTensor:
		"""Linear classification head for multilabel task. During inference, only `x` is given so we compute `y_hat`
		only.

		Args:
			x (InputTensor):
				`x` contains either the input data or the predictions from previous step.

		Returns:
			y_hat (OutputTensor):
				The final prediction.
		"""
		if isinstance(x, tuple):
			x = x[-1]
		x = self.fc(x)
		if isinstance(x, list):
			x = sum(x) / float(len(x))
		y_hat = x
		
		# NOTE: Calculate class-score (softmax)
		y_hat = F.sigmoid(y_hat) if y_hat is not None else None

		return y_hat
