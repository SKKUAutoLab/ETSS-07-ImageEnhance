#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MNASNet backbones.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
from torch import nn
from torchvision.models.mnasnet import _get_depths
from torchvision.models.mnasnet import _stack

from torchkit.core.utils import Indexes
from torchkit.models.builder import BACKBONES
from torchkit.models.builder import CLASSIFIERS
from torchkit.models.builder import MODELS
from .image_classifier import ImageClassifier

logger = logging.getLogger()


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum
# is 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


# MARK: - MNASNet

# noinspection PyUnusedLocal,PyDefaultArgument,PyMethodOverriding
@BACKBONES.register(name="mnasnet")
@CLASSIFIERS.register(name="mnasnet")
@MODELS.register(name="mnasnet")
class MNASNet(ImageClassifier):
	"""MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
	implements the B1 variant of the model.
	
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
			Name of the backbone. Default: `mnasnet`.
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
	
	# Version 2 adds depth scaling in the initial stages of the network.
	_version = 2
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		cfg        : Union[str, list, dict] = dict(alpha=0.5, dropout=0.2),
		name       : Optional[str]    	  	= "mnasnet",
		num_classes: Optional[int] 	  	    = None,
		out_indexes: Indexes		  	    = -1,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			name=name, num_classes=num_classes, out_indexes=out_indexes,
			pretrained=pretrained, *args, **kwargs
		)
		# NOTE: Get Hyperparameters
		assert isinstance(cfg, dict)
		self.cfg = cfg
		
		alpha   = cfg["alpha"]
		dropout = cfg["dropout"]
		assert alpha > 0.0
		
		# NOTE: Features
		self.layers = self.features = self.create_features(alpha)
		
		# NOTE: Classifier
		self.classifier = self.create_classifier(dropout, self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
		else:
			self.initialize_weights()
	
	# MARK: Configure
	
	@staticmethod
	def create_features(alpha: float) -> nn.Sequential:
		depths = _get_depths(alpha)
		layers = [
			# First layer: regular conv.
			nn.Conv2d(3, depths[0], (3, 3), (2, 2), padding=1, bias=False),
			nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
			nn.ReLU(inplace=True),
			# Depthwise separable, no skip.
			nn.Conv2d(depths[0], depths[0], (3, 3), (1, 1), padding=1,
					  groups=depths[0], bias=False),
			nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
			nn.ReLU(inplace=True),
			nn.Conv2d(depths[0], depths[1], (1, 1), (1, 1), padding=0,
					  bias=False),
			nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM),
			# MNASNet blocks: stacks of inverted residuals.
			_stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
			_stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
			_stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
			_stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
			_stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
			_stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
			# Final mapping to classifier input.
			nn.Conv2d(depths[7], 1280, (1, 1), (1, 1), padding=0, bias=False),
			nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
			nn.ReLU(inplace=True),
		]
		return nn.Sequential(*layers)
	
	@staticmethod
	def create_classifier(
		dropout: float, num_classes: Optional[int]
	) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Sequential(
				nn.Dropout(p=dropout, inplace=True),
				nn.Linear(1280, num_classes)
			)
		else:
			classifier = nn.Identity()
		return classifier
	
	def initialize_weights(self) -> None:
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out",
										nonlinearity="relu")
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight, mode="fan_out",
										 nonlinearity="sigmoid")
				nn.init.zeros_(m.bias)
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		x = self.layers(x)
		# Equivalent to global avgpool and removing H and W dimensions.
		x = x.mean([2, 3])
		x = self.classifier(x)
		return x
		

# MARK: - MNASNet0_5

@BACKBONES.register(name="mnasnet_x0.5")
@CLASSIFIERS.register(name="mnasnet_x0.5")
@MODELS.register(name="mnasnet_x0.5")
class MNASNet_x0_5(MNASNet):
	"""MNASNet with depth multiplier of 0.5 from `MnasNet: Platform-Aware
	Neural Architecture Search for Mobile -
	<https://arxiv.org/pdf/1807.11626.pdf>`.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
			file_name="mnasnet_x0.5_imagenet.pth", num_classes=1000,
		),
	}

	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "mnasnet_x0.5",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg         = dict(alpha=0.5, dropout=0.2),
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - MNASNet0_75

@BACKBONES.register(name="mnasnet_x0.75")
@CLASSIFIERS.register(name="mnasnet_x0.75")
@MODELS.register(name="mnasnet_x0.75")
class MNASNet_x0_75(MNASNet):
	"""MNASNet with depth multiplier of 0.75 from `MnasNet: Platform-Aware
	Neural Architecture Search for Mobile -
    <https://arxiv.org/pdf/1807.11626.pdf>`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "mnasnet_x0.75",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg         = dict(alpha=0.75, dropout=0.2),
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
			

# MARK: - MNASNet1_0

@BACKBONES.register(name="mnasnet_x1.0")
@CLASSIFIERS.register(name="mnasnet_x1.0")
@MODELS.register(name="mnasnet_x1.0")
class MNASNet_x1_0(MNASNet):
	"""MNASNet with depth multiplier of 1.0 from `MnasNet: Platform-Aware
	Neural Architecture Search for Mobile -
    <https://arxiv.org/pdf/1807.11626.pdf>`.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
			file_name="mnasnet_x1.0_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "mnasnet_x1.0",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg         = dict(alpha=1.0, dropout=0.2),
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - MNASNet1_3

@BACKBONES.register(name="mnasnet_x1.3")
@CLASSIFIERS.register(name="mnasnet_x1.3")
@MODELS.register(name="mnasnet_x1.3")
class MNASNet_x1_3(MNASNet):
	"""MNASNet with depth multiplier of 1.3 from `MnasNet: Platform-Aware
	Neural Architecture Search for Mobile -
    <https://arxiv.org/pdf/1807.11626.pdf>`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "mnasnet_x1.3",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg         = dict(alpha=1.3, dropout=0.2),
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
