#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet backbones.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition

from torchkit.core.utils import Indexes
from torchkit.models.builder import BACKBONES
from torchkit.models.builder import CLASSIFIERS
from torchkit.models.builder import MODELS
from .image_classifier import ImageClassifier

logger = logging.getLogger()


cfgs = {
	"densenet121": dict(
		growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
		bn_size=4, drop_rate=0, memory_efficient=False
	),
	"densenet161": dict(
		growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96,
		bn_size=4, drop_rate=0, memory_efficient=False
	),
	"densenet169": dict(
		growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64,
		bn_size=4, drop_rate=0, memory_efficient=False
	),
	"densenet201": dict(
		growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64,
		bn_size=4, drop_rate=0, memory_efficient=False
	),
}


# MARK: - DenseNet

# noinspection PyMethodOverriding
@BACKBONES.register(name="densenet")
@CLASSIFIERS.register(name="densenet")
@MODELS.register(name="densenet")
class DenseNet(ImageClassifier):
	"""DenseNet backbone.
	
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
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		cfg        : Union[str, list, dict],
		name       : Optional[str]    	  	= "densenet",
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
		if isinstance(cfg, str) and cfg in cfgs:
			cfg = cfgs[cfg]
		assert isinstance(cfg, dict)
		self.cfg = cfg
		
		# NOTE: Features
		self.features, num_features = self.create_features(cfg)
		
		# NOTE: Classifier
		self.classifier = self.create_classifier(num_features, self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
		else:
			self.initialize_weights()
	
	# MARK: Configure
	
	@staticmethod
	def create_features(cfg: dict) -> tuple[nn.Sequential, int]:
		growth_rate 	  = cfg["growth_rate"]
		# How many filters to add each layer (`k` in paper).
		block_config	  = cfg["block_config"]
		# How many layers in each pooling block.
		num_init_features = cfg["num_init_features"]
		# The number of filters to learn in the first convolution layer.
		bn_size    		  = cfg["bn_size"]
		# Multiplicative factor for number of bottle neck layers (i.e.
		# bn_size * k features in the bottleneck layer). Default: `4`.
		drop_rate  		  = cfg["drop_rate"]
		# Dropout rate after each dense layer. Default: `0`.
		memory_efficient  = cfg["memory_efficient"]
		# If True, uses checkpointing. Much more memory efficient, but slower.
		# Default: `False`. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`.
		
		# First convolution
		features = nn.Sequential(OrderedDict([
			("conv0", nn.Conv2d(3, num_init_features, kernel_size=(7, 7),
								stride=(2, 2), padding=3, bias=False)),
			("norm0", nn.BatchNorm2d(num_init_features)),
			("relu0", nn.ReLU(inplace=True)),
			("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))
		
		# Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(
				num_layers         = num_layers,
				num_input_features = num_features,
				bn_size            = bn_size,
				growth_rate        = growth_rate,
				drop_rate          = drop_rate,
				memory_efficient   = memory_efficient
			)
			features.add_module("denseblock%d" % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features,
									num_output_features=num_features // 2)
				features.add_module("transition%d" % (i + 1), trans)
				num_features = num_features // 2
		
		# Final batch norm
		features.add_module("norm5", nn.BatchNorm2d(num_features))
		return features, num_features
	
	@staticmethod
	def create_classifier(
		num_features: int, num_classes: Optional[int]
	) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Linear(num_features, num_classes)
		else:
			classifier = nn.Identity()
		return classifier
	
	def initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = F.relu(x, inplace=True)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


# MARK: - DenseNet121

@BACKBONES.register(name="densenet121")
@CLASSIFIERS.register(name="densenet121")
@MODELS.register(name="densenet121")
class DenseNet121(DenseNet):
	"""Densenet-121 model from `Densely Connected Convolutional Networks -
	<https://arxiv.org/pdf/1608.06993.pdf>`. The required minimum input size of
	the model is 29x29.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/densenet121-a639ec97.pth",
			file_name="densenet121_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "densenet121",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="densenet121", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)
	

# MARK: - DenseNet161

@BACKBONES.register(name="densenet161")
@CLASSIFIERS.register(name="densenet161")
@MODELS.register(name="densenet161")
class DenseNet161(DenseNet):
	"""Densenet-161 model from `"Densely Connected Convolutional Networks"
	<https://arxiv.org/pdf/1608.06993.pdf>`. The required minimum input size of
	the model is 29x29.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/densenet161-8d451a50.pth",
			file_name="densenet161_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "densenet161",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="densenet161", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)
			

# MARK: - DenseNet169

@BACKBONES.register(name="densenet169")
@CLASSIFIERS.register(name="densenet169")
@MODELS.register(name="densenet169")
class DenseNet169(DenseNet):
	"""Densenet-169 model from `Densely Connected Convolutional Networks -
	<https://arxiv.org/pdf/1608.06993.pdf>`. The required minimum input size of
	the model is 29x29.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/densenet169-b2777c0a.pth",
			file_name="densenet169_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "densenet169",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="densenet169", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)


# MARK: - DenseNet201

@BACKBONES.register(name="densenet201")
@CLASSIFIERS.register(name="densenet201")
@MODELS.register(name="densenet201")
class DenseNet201(DenseNet):
	"""Densenet-201 model from `Densely Connected Convolutional Networks
	<https://arxiv.org/pdf/1608.06993.pdf>`. The required minimum input size of
	the model is 29x29.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/densenet201-c1103571.pth",
			file_name="densenet201_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	  	= "densenet201",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="densenet201", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)
