#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG backbones.
"""

from __future__ import annotations

import logging
from typing import cast
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


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
		  "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
		  512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
		  512, "M", 512, 512, 512, 512, "M"],
}


# MARK: - VGG

# noinspection PyMethodOverriding
@BACKBONES.register(name="vgg")
@CLASSIFIERS.register(name="vgg")
@MODELS.register(name="vgg")
class VGG(ImageClassifier):
	"""VGG.
	
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
		batch_norm (bool):
			Should use batch norm layer? Default: `False`.
	
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
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		cfg		   : list[Union[str, int]],
		batch_norm : bool			    	= False,
		name       : Optional[str]    	    = "vgg",
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
		self.features = self.create_features(cfg, batch_norm)
		
		# NOTE: Head (pool + classifier)
		self.avgpool 	= nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = self.create_classifier(self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
		else:
			self.initialize_weights()
		
	# MARK: Configure
	
	@staticmethod
	def create_features(cfg, batch_norm) -> nn.Sequential:
		layers: list[nn.Module] = []
		in_channels = 3
		for v in cfg:
			if v == "M":
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				v = cast(int, v)
				conv2d = nn.Conv2d(
					in_channels, v, kernel_size=(3, 3), padding=1
				)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)
	
	@staticmethod
	def create_classifier(num_classes: Optional[int]) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Sequential(
				nn.Linear(512 * 7 * 7, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096, num_classes),
			)
		else:
			classifier = nn.Identity()
		return classifier
	
	def initialize_weights(self) -> None:
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(
					m.weight, mode="fan_out", nonlinearity="relu"
				)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
	

# MARK: - VGG11

@BACKBONES.register(name="vgg11")
@CLASSIFIERS.register(name="vgg11")
@MODELS.register(name="vgg11")
class VGG11(VGG):
	"""VGG 11-layer model (configuration "A") from `Very Deep Convolutional
	Networks For Large-Scale Image Recognition -
	<https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size of
	the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg11-8a719046.pth",
			file_name="vgg11_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg11",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["A"],
			batch_norm  = False,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - VGG11Bn

@BACKBONES.register(name="vgg11_bn")
@CLASSIFIERS.register(name="vgg11_bn")
@MODELS.register(name="vgg11_bn")
class VGG11Bn(VGG):
	"""VGG 11-layer model (configuration "A") from `Very Deep Convolutional
	Networks For Large-Scale Image Recognition -
	<https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size of
	the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
			file_name="vgg11_bn_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg11_bn",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["A"],
			batch_norm  = True,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
			

# MARK: - VGG13

@BACKBONES.register(name="vgg13")
@CLASSIFIERS.register(name="vgg13")
@MODELS.register(name="vgg13")
class VGG13(VGG):
	"""VGG 13-layer model (configuration "B") `Very Deep Convolutional
	Networks For Large-Scale Image Recognition
	- <https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size
	of the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg13-19584684.pth",
			file_name="vgg13_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg13",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["B"],
			batch_norm  = False,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - VGG13Bn

@BACKBONES.register(name="vgg13_bn")
@CLASSIFIERS.register(name="vgg13_bn")
@MODELS.register(name="vgg13_bn")
class VGG13Bn(VGG):
	"""VGG 13-layer model (configuration "B") `Very Deep Convolutional
	Networks For Large-Scale Image Recognition
	- <https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size
	of the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
			file_name="vgg13_bn_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg13_bn",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["B"],
			batch_norm  = True,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
		

# MARK: - VGG16

@BACKBONES.register(name="vgg16")
@CLASSIFIERS.register(name="vgg16")
@MODELS.register(name="vgg16")
class VGG16(VGG):
	"""VGG 16-layer model (configuration "D") `Very Deep Convolutional Networks
	For Large-Scale Image Recognition - <https://arxiv.org/pdf/1409.1556.pdf>`.
    The required minimum input size of the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg16-397923af.pth",
			file_name="vgg16_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg16",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["D"],
			batch_norm  = False,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - VGG16Bn

@BACKBONES.register(name="vgg16_bn")
@CLASSIFIERS.register(name="vgg16_bn")
@MODELS.register(name="vgg16_bn")
class VGG16Bn(VGG):
	"""VGG 16-layer model (configuration "D") `Very Deep Convolutional Networks
	For Large-Scale Image Recognition - <https://arxiv.org/pdf/1409.1556.pdf>`.
    The required minimum input size of the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
			file_name="vgg16_bn_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg16_bn",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["D"],
			batch_norm  = True,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
		
		
# MARK: - VGG19

@BACKBONES.register(name="vgg19")
@CLASSIFIERS.register(name="vgg19")
@MODELS.register(name="vgg19")
class VGG19(VGG):
	"""VGG 19-layer model (configuration "E") `Very Deep Convolutional
	Networks For Large-Scale Image Recognition -
	<https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size
	of the model is 32x32.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
			file_name="vgg19_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg19",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["E"],
			batch_norm  = False,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)


# MARK: - VGG19Bn

@BACKBONES.register(name="vgg19_bn")
@CLASSIFIERS.register(name="vgg19_bn")
@MODELS.register(name="vgg19_bn")
class VGG19Bn(VGG):
	"""VGG 19-layer model (configuration "E") `Very Deep Convolutional
	Networks For Large-Scale Image Recognition -
	<https://arxiv.org/pdf/1409.1556.pdf>`. The required minimum input size
	of the model is 32x32.
	"""

	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
			file_name="vgg19_bn_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "vgg19_bn",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg		    = cfgs["E"],
			batch_norm  = True,
			out_indexes = out_indexes,
			name        = name,
			num_classes = num_classes,
			pretrained  = pretrained,
			*args, **kwargs
		)
