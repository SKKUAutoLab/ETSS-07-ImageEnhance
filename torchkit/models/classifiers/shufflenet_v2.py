#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ShuffleNetV2 backbones.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
from torch import nn
from torchvision.models.shufflenetv2 import InvertedResidual

from torchkit.core.utils import Indexes
from torchkit.models.builder import BACKBONES
from torchkit.models.builder import CLASSIFIERS
from torchkit.models.builder import MODELS
from .image_classifier import ImageClassifier

logger = logging.getLogger()


cfgs = {
	"shufflenet_v2_x0_5": dict(
		stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024],
		inverted_residual=InvertedResidual
	),
	"shufflenet_v2_x1_0": dict(
		stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024],
		inverted_residual=InvertedResidual
	),
	"shufflenet_v2_x1_5": dict(
		stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024],
		inverted_residual=InvertedResidual
	),
	"shufflenet_v2_x2_0": dict(
		stages_repeats=[4, 8, 4], stages_out_channels=[24, 244, 488, 976, 2048],
		inverted_residual=InvertedResidual
	),
}


# MARK: - ShuffleNetV2

# noinspection PyMethodOverriding
@BACKBONES.register(name="shufflenet_v2")
@CLASSIFIERS.register(name="shufflenet_v2")
@MODELS.register(name="shufflenet_v2")
class ShuffleNetV2(ImageClassifier):
	"""ShuffleNetV2.
	
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
			Name of the backbone. Default: `shufflenet_v2`.
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
		cfg		   : Union[str, list, dict],
		name       : Optional[str]    	    = "shufflenet_v2",
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
		
		stages_repeats 	    = cfg["stages_repeats"]
		stages_out_channels = cfg["stages_out_channels"]
		inverted_residual   = cfg["inverted_residual"]
		
		if len(stages_repeats) != 3:
			raise ValueError("Expected stages_repeats as list of 3 positive "
							 "ints")
		if len(stages_out_channels) != 5:
			raise ValueError("Expected stages_out_channels as list of 5 "
							 "positive ints")
		self._stage_out_channels = stages_out_channels
		
		input_channels  = 3
		output_channels = self._stage_out_channels[0]
		
		# NOTE: Features
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, (3, 3), (2, 2), 1,
					  bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)
		input_channels = output_channels
		
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		
		# Static annotations for mypy
		self.stage2: nn.Sequential
		self.stage3: nn.Sequential
		self.stage4: nn.Sequential
		stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
		for name, repeats, output_channels in zip(
			stage_names, stages_repeats, self._stage_out_channels[1:]
		):
			seq = [inverted_residual(input_channels, output_channels, 2)]
			for i in range(repeats - 1):
				seq.append(inverted_residual(
					output_channels, output_channels, 1
				))
			setattr(self, name, nn.Sequential(*seq))
			input_channels = output_channels
		
		output_channels = self._stage_out_channels[-1]
		self.conv5 = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, (1, 1), (1, 1), 0,
					  bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(inplace=True),
		)
		
		# NOTE: Head (Pool + Classifier layer)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = self.create_classifier(output_channels, self.num_classes)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
	
		# NOTE: Alias
		self.features = nn.Sequential(
			self.conv1, self.maxpool, self.stage2, self.stage3, self.stage4,
			self.conv5
		)
		self.classifier = self.fc
	
	# MARK: Configure
	
	@staticmethod
	def create_classifier(
		num_features: int, num_classes: Optional[int]
	) -> nn.Module:
		if num_classes and num_classes > 0:
			classifier = nn.Linear(num_features, num_classes)
		else:
			classifier = nn.Identity()
		return classifier
	
	# MARK: Forward Pass
	
	def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
		return self._forward_impl(x)
	
	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		# See note [TorchScript super()]
		x = self.conv1(x)
		x = self.maxpool(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.conv5(x)
		x = x.mean([2, 3])  # globalpool
		x = self.fc(x)
		return x


# MARK: - ShuffleNetV2_x0_5

@BACKBONES.register(name="shufflenet_v2_x0_5")
@CLASSIFIERS.register(name="shufflenet_v2_x0_5")
@MODELS.register(name="shufflenet_v2_x0_5")
class ShuffleNetV2_x0_5(ShuffleNetV2):
	"""ShuffleNetV2 with 0.5x output channels, as described in `ShuffleNet V2:
	Practical Guidelines for Efficient CNN Architecture Design -
	<https://arxiv.org/abs/1807.11164>`.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
			file_name="shufflenet_v2_x0_5_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "shufflenet_v2_x0_5",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="shufflenet_v2_x0_5", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)


# MARK: - ShuffleNetV2_x1_0

@BACKBONES.register(name="shufflenet_v2_x1_0")
@CLASSIFIERS.register(name="ShuffleNetV2_x1_0")
@MODELS.register(name="ShuffleNetV2_x1_0")
class ShuffleNetV2_x1_0(ShuffleNetV2):
	"""ShuffleNetV2 with 1.0x output channels, as described in `ShuffleNet V2:
	Practical Guidelines for Efficient CNN Architecture Design -
	<https://arxiv.org/abs/1807.11164>`.
	"""
	
	model_zoo = {
		"imagenet": dict(
			path="https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
			file_name="shufflenet_v2_x1_0_imagenet.pth", num_classes=1000,
		),
	}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "shufflenet_v2_x1_0",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="shufflenet_v2_x1_0", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)
			

# MARK: - ShuffleNetV2_x1_5

@BACKBONES.register(name="shufflenet_v2_x1_5")
@CLASSIFIERS.register(name="shufflenet_v2_x1_5")
@MODELS.register(name="shufflenet_v2_x1_5")
class ShuffleNetV2_x1_5(ShuffleNetV2):
	"""ShuffleNetV2 with 1.5x output channels, as described in `ShuffleNet V2:
	Practical Guidelines for Efficient CNN Architecture Design -
	<https://arxiv.org/abs/1807.11164>`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "shufflenet_v2_x1_5",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="shufflenet_v2_x1_5", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)


# MARK: - ShuffleNetV2_x2_0

@BACKBONES.register(name="shufflenet_v2_x2_0")
@CLASSIFIERS.register(name="shufflenet_v2_x2_0")
@MODELS.register(name="shufflenet_v2_x2_0")
class ShuffleNetV2_x2_0(ShuffleNetV2):
	"""ShuffleNetV2 with 2.0x output channels, as described in `ShuffleNet V2:
	Practical Guidelines for Efficient CNN Architecture Design -
	<https://arxiv.org/abs/1807.11164>`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes		  	    = -1,
		name       : Optional[str]    	    = "shufflenet_v2_x2_0",
		num_classes: Optional[int] 	  	    = None,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="shufflenet_v2_x2_0", out_indexes=out_indexes, name=name,
			num_classes=num_classes, pretrained=pretrained, *args, **kwargs
		)
