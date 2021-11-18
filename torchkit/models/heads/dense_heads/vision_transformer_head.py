# ==================================================================== #
# File name: vision_transformer_head.py
# Author: Long H. Pham
# Date created: 09/02/2021
# The `torchkit.models.heads.dense_heads.vision_transformer_head` defines
# the vision Transformer classifier head.
# ==================================================================== #
from __future__ import annotations

import logging
from typing import Optional
from typing import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import InputTensor
from torchkit.core.utils import OutputTensor
from torchkit.models.builder import ACT_LAYERS
from torchkit.models.builder import HEADS
from .cls_head import ClsHead

logger = logging.getLogger()


# MARK: - VisionTransformerClsHead

# noinspection PyDefaultArgument
@HEADS.register(name="vision_transformer_cls_head")
@HEADS.register(name="VisionTransformerClsHead")
class VisionTransformerClsHead(ClsHead):
	"""Vision Transformer classifier head.
	
	Attributes:
		num_classes (int):
			Number of categories excluding the background category.
		in_channels (int):
			Number of channels in the input feature map.
		hidden_dim (int, optional):
			Number of the dimensions for hidden layer. Only available during pre-training. Default: `None`.
		act_cfg (dict):
			The activation config. Only available during pre-training. Default: `dict(name="Tanh")`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		num_classes: int,
		in_channels: int,
		hidden_dim : Optional[int] = None,
		act_cfg    : dict          = dict(name="Tanh"),
		name       : Optional[str] = "vision_transformer_cls_head",
		*args, **kwargs
	):
		"""
		
		Args:
			num_classes (int):
				Number of categories excluding the background category.
			in_channels (int):
				Number of channels in the input feature map.
			hidden_dim (int, optional):
				Number of the dimensions for hidden layer. Only available during pre-training. Default: `None`.
			act_cfg (dict):
				The activation config. Only available during pre-training. Default: `dict(name="Tanh")`.
			name (str, optional):
				Name of the head. Default: `vision_transformer_cls_head`.
		"""
		super().__init__(
			name = name,
			*args, **kwargs
		)
		self.in_channels = in_channels
		self.num_classes = num_classes
		self.hidden_dim  = hidden_dim
		self.act_cfg     = act_cfg

		if self.num_classes <= 0:
			raise ValueError(f"`num_classes`={num_classes} must be a positive integer.")

		self._init_layers()
	
	# MARK: Configure
	
	def _init_layers(self):
		if self.hidden_dim is None:
			layers = [("head", nn.Linear(self.in_channels, self.num_classes))]
		else:
			layers = [
				("pre_logits", nn.Linear(self.in_channels, self.hidden_dim)),
				("act", ACT_LAYERS.build_from_cfg(self.act_cfg)),
				("head",       nn.Linear(self.hidden_dim, self.num_classes)),
			]
		self.layers = nn.Sequential(OrderedDict(layers))

	def init_weights(self):
		super(VisionTransformerClsHead, self).init_weights()
		# Modified from ClassyVision
		if hasattr(self.layers, "pre_logits"):
			# Lecun norm
			kaiming_init(module=self.layers.pre_logits, mode="fan_in", nonlinearity="linear")
		constant_init(module=self.layers.head, val=0)
	
	# MARK: Forward Pass
	
	def forward_xy(self, x: InputTensor, y: InputTensor, *args, **kwargs) -> ForwardXYOutput:
		"""Vision Transformer classifier head. Both `x` and `y` are given, hence, we compute the loss and metrics also.

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
		y_hat = self.layers(x)
		
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
		y_hat = self.layers(x)
		if isinstance(y_hat, (list, tuple)):
			y_hat = sum(y_hat) / float(len(y_hat))
		
		# NOTE: Calculate class-score (softmax)
		y_hat = F.softmax(y_hat, dim=1) if y_hat is not None else None

		return y_hat
