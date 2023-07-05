#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Classifier.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

from torchkit.core.image import imshow_cls_plt
from torchkit.core.runner import BaseModel
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import Images
from torchkit.core.utils import Indexes
from torchkit.core.utils import Tensors
from torchkit.core.utils import to_4d_array

logger = logging.getLogger()


# MARK: - ImageClassifier

class ImageClassifier(BaseModel, metaclass=ABCMeta):
	"""Image Classifier is a base class for image classification models. It
	extends the BaseModel class.
	
	Usually, we add functions such as:
	- Define main components.
	- Forward pass with custom loss and metrics.
	- Result visualization.
	
	Attributes:
		features (nn.Module):
			The feature extraction module.
		classifier (nn.Module):
			The classifier head.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.features  : Optional[nn.Module] = None
		self.classifier: Optional[nn.Module] = None
		
	# MARK: Properties
	
	@property
	def with_features(self) -> bool:
		"""Return whether if the `features` has been defined."""
		return hasattr(self, "features") and self.features is not None
	
	@property
	def with_classifier(self) -> bool:
		"""Check if the `classifier` attribute has been defined."""
		return hasattr(self, "classifier") and self.classifier is not None
	
	# MARK: Forward Pass
	
	def forward_train(
		self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs
	) -> ForwardXYOutput:
		"""Forward pass during training with both `x` and `y` are given.

		Args:
			x (torch.Tensor):
				The input image of shape [B, C, H, W].
			y (torch.Tensor):
				The ground-truth label of each input.

		Returns:
			y_hat (torch.Tensor):
				The final predictions.
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		# NOTE: By default, call `forward_infer()` and calculate metrics
		y_hat   = self.forward_infer(x=x, *args, **kwargs)
		metrics = {}
		if self.with_loss:
			metrics["loss"] = self.loss(y_hat, y)
		if self.with_metrics:
			ms 	     = {m.name: m(y_hat, y) for m in self.metrics}
			metrics |= ms  # NOTE: 3.9+ ONLY
		metrics = metrics if len(metrics) else None
		return y_hat, metrics
		
	@abstractmethod
	def forward_infer(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		"""Forward pass during inference with only `x` is given.

		Args:
			x (torch.Tensor):
				The input image of shape [B, C, H, W].

		Returns:
			y_hat (torch.Tensor):
				The final predictions.
		"""
		pass
	
	def forward_features(
		self, x: torch.Tensor, out_indexes: Optional[Indexes] = None
	) -> Tensors:
		"""Forward pass for features extraction.
		
		Args:
			x (torch.Tensor):
				The input image.
			out_indexes (Indexes, optional):
				The list of layers' indexes to extract features. This is called
				in `forward_features()` and is useful when the model
				is used as a component in another model.
				- If is a `tuple` or `list`, return an array of features.
				- If is a `int`, return only the feature from that layer's
				  index.
				- If is `-1`, return the last layer's output.
				Default: `None`.
		"""
		out_indexes = self.out_indexes if out_indexes is None else out_indexes
		assert self.with_features, f"`features` has not been defined."
		
		y_hat = []
		for idx, m in enumerate(self.features.children()):
			x = m(x)
			if isinstance(out_indexes, (tuple, list)) and (idx in out_indexes):
				y_hat.append(x)
			elif isinstance(out_indexes, int) and (idx == out_indexes):
				return x
			else:
				y_hat = x
		return y_hat
	
	# MARK: Visualization
	
	def show_results(
		self,
		x            : Images,
		y            : Optional[Images] = None,
		y_hat        : Optional[Images] = None,
		filepath     : Optional[str]    = None,
		image_quality: int              = 95,
		show         : bool             = False,
		show_max_n   : int              = 8,
		wait_time    : float            = 0.01,
		*args, **kwargs
	):
		"""Draw `result` over input image.

		Args:
			x (Images):
				The input images.
			y (Images, optional):
				The ground truth.
			y_hat (Images, optional):
				The predictions.
			filepath (str, optional):
				The file path to save the debug result.
			image_quality (int):
				The image quality to be saved. Default: `95`.
			show (bool):
				If `True` shows the results on the screen. Default: `False`.
			show_max_n (int):
				Maximum debugging items to be shown. Default: `8`.
			wait_time (float):
				Pause some times before showing the next image.
		"""
		results  = to_4d_array(x)
		y		 = y.detach().cpu().numpy()
		y_hat    = y_hat.detach().cpu().numpy()

		filepath = self.debug_image_filepath if filepath is None else filepath
		save_cfg = dict(filepath=filepath,
						pil_kwargs=dict(quality=image_quality))
		imshow_cls_plt(images=results, preds=y_hat, gts=y,
					   classlabels=self.classlabels, top_k=5, scale=2,
					   save_cfg=save_cfg, show=show, show_max_n=show_max_n,
					   wait_time=wait_time)
