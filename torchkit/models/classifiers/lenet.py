#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LeNet backbone.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from torchkit.core.utils import Indexes
from torchkit.models.builder import BACKBONES
from torchkit.models.builder import CLASSIFIERS
from torchkit.models.builder import MODELS
from .image_classifier import ImageClassifier

logger = logging.getLogger()


# MARK: - LeNet5

# noinspection PyMethodOverriding
@BACKBONES.register(name="lenet")
@BACKBONES.register(name="lenet5")
@CLASSIFIERS.register(name="lenet")
@CLASSIFIERS.register(name="lenet5")
@MODELS.register(name="lenet")
@MODELS.register(name="lenet5")
class LeNet5(ImageClassifier):
	"""`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`. The input for LeNet-5
	is a 32x32 grayscale image.
	
	Attributes:
		cfg (str, list, dict, optional):
			The config to build the model's layers.
			- If `str`, use the corresponding config from the predefined
			  config dict. This is used to build the model dynamically.
			- If a file or filepath, it leads to the external config file that
			  is used to build the model dynamically.
			- If `list`, then each element in the list is the corresponding
			  config for each layer in the model. This is used to build the
			  model dynamically.
			- If `dict`, it usually contains the hyperparameters used to
			  build the model manually in the code.
			- If `None`, then you should manually define the model.
			Remark: You have 5 ways to build the model, so choose the style
			that you like.
			
	Args:
		name (str, optional):
			Name of the backbone. Default: `alexnet`.
		num_classes (int, optional):
			Number of classes for classification. Default: `None`.
		out_indexes (Indexes):
			The list of output tensors taken from specific layers' indexes.
			If `>= 0`, return the ith layer's output.
			If `-1`, return the final layer's output. Default: `-1`.
		pretrained (bool, str):
			Use pretrained weights. If `True`, returns a model pre-trained on
			ImageNet. If `str`, load weights from saved file. Default: `True`.
			- If `True`, returns a model pre-trained on ImageNet.
			- If `str` and is a weight file(path), then load weights from
			  saved file.
			- In each inherited model, `pretrained` can be a dictionary's
			  key to get the corresponding local file or url of the weight.
	"""
	
	model_zoo = {}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		name       : Optional[str]    	    = "lenet",
		num_classes: Optional[int] 	  	    = None,
		out_indexes: Indexes		  	    = -1,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			name=name, num_classes=num_classes, out_indexes=out_indexes,
			pretrained=pretrained, *args, **kwargs
		)
		
		# NOTE: Features
		self.features = nn.Sequential(
			nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=(2, 2)),
			nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=(2, 2)),
			nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1)),
			nn.Tanh(),
		)
		
		# NOTE: Head (classifier)
		self.classifier = self.create_classifier(self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
	
	# MARK: Configure
	
	@staticmethod
	def create_classifier(num_classes: Optional[int]) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Sequential(
				nn.Linear(120, 84),
				nn.Tanh(),
				nn.Linear(84, num_classes),
			)
		else:
			classifier = nn.Identity()
		return classifier
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.classifier(x)
		return x
