#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet Backbone.

Paper <https://en.wikipedia.org/wiki/AlexNet>.

Reimplement from `torchvision.models.alexnet`.
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


# MARK: - AlexNet

# noinspection PyMethodOverriding
@BACKBONES.register(name="alexnet")
@CLASSIFIERS.register(name="alexnet")
@MODELS.register(name="alexnet")
class AlexNet(ImageClassifier):
	"""`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_. The required minimum
	input size of the model is 63x63.
	
	Args:
		name (str, optional):
			Name of the model. Default: `alexnet`.
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
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
			file_name="alexnet_owt_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		name       : Optional[str]    	    = "alexnet",
		num_classes: Optional[int] 	  	    = None,
		out_indexes: Indexes 			    = -1,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			name=name, num_classes=num_classes, out_indexes=out_indexes,
			pretrained=pretrained, *args, **kwargs
		)
		
		# NOTE: Features
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		
		# NOTE: Head (pool + classifier)
		self.avgpool 	= nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = self.create_classifier(self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
	
	# MARK: Configure
	
	@staticmethod
	def create_classifier(num_classes: Optional[int]) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(256 * 6 * 6, 4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(inplace=True),
				nn.Linear(4096, num_classes),
			)
		else:
			classifier = nn.Identity()
		return classifier
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
