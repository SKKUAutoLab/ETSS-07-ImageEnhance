#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RetinexNet.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from torchkit.core.image import imshow_plt
from torchkit.core.metric import METRICS
from torchkit.core.runner import BaseModel
from torchkit.core.utils import Arrays
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import Images
from torchkit.core.utils import Indexes
from torchkit.core.utils import Size2T
from torchkit.core.utils import Tensors
from torchkit.core.utils import to_2tuple
from torchkit.core.utils import to_4d_array
from torchkit.models.builder import ENHANCERS
from torchkit.models.builder import MODELS
from torchkit.models.losses import DecomLoss
from torchkit.models.losses import EnhanceLoss
from torchkit.models.losses import RetinexLoss

logger = logging.getLogger()

__all__ = ["DecomNet", "EnhanceNet", "EnhanceUNet", "RetinexNet", "Phase"]


# MARK: - DecomNet

class DecomNet(nn.Module):
	"""DecomNet is one of the two sub-networks used in RetinexNet model.
	DecomNet breaks the RGB image into a 1-channel intensity map and a
	3-channels reflectance map.

	Attributes:
		num_activation_layers (int):
			Number of activation layers. Default: `5`.
		channels (int):
			The number of output channels (or filters) for the `Conv2D` layer
			in the decomnet. Default: `64`.
		kernel_size (Size2T):
			The kernel size for the `Conv2D` layer in the decomnet.
			Default: `3`.
		use_batchnorm (bool):
			If `True`, use Batch Normalization layer between `Conv2D` and
			`Activation` layers. Default: `True`.
		name (str):
			Name of the backbone. Default: `decomnet`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		num_activation_layers: int    = 5,
		channels             : int    = 64,
		kernel_size          : Size2T = 3,
		use_batchnorm        : bool   = True,
		*args, **kwargs
	):
		super().__init__()
		self.name 	  	   = "decomnet"
		self.use_batchnorm = use_batchnorm
		
		channels      = channels
		kernel_size   = to_2tuple(kernel_size)
		kernel_size_3 = tuple([i * 3 for i in kernel_size])
		
		convs = []
		# Shallow feature_extractor extraction
		convs.append(nn.Conv2d(4, channels, kernel_size_3, padding=4,
							   padding_mode="replicate"))
		# Activation layers
		for i in range(num_activation_layers):
			convs.append(nn.Conv2d(channels, channels, kernel_size, padding=1,
								   padding_mode="replicate"))
			convs.append(nn.ReLU(inplace=True))
		self.features = nn.Sequential(*convs)
		
		# Reconstruction layer
		self.recon = nn.Conv2d(channels, 4, kernel_size, padding=1,
							   padding_mode="replicate")
		self.bn = nn.BatchNorm2d(4)
	
	# MARK: Forward Pass
	
	def forward(self, x: torch.Tensor) -> Tensors:
		"""Forward pass.

		Args:
			x (torch.Tensor):
				The input images.

		Returns:
			y_hat (torch.Tensor):
				A single tensor of shape [B, 4, H, W], consisting of
				reflectance maps [:, 0:3, :, :] and illumination maps
				[:, 3:4, :, :].
		"""
		x_max = torch.max(input=x, dim=1, keepdim=True)[0]
		x_cat = torch.cat(tensors=(x_max, x), dim=1)
		x     = self.features(x_cat)
		y_hat = self.recon(x)
		if self.use_batchnorm:
			y_hat = self.bn(y_hat)
		r = torch.sigmoid(y_hat[:, 0:3, :, :])
		i = torch.sigmoid(y_hat[:, 3:4, :, :])
		return r, i
	

# MARK: - EnhanceNet

# noinspection PyMethodOverriding
class EnhanceNet(nn.Module):
	"""EnhanceNet is one of the two sub-networks used in RetinexNet model.
	EnhanceNet increases the light distribution in the 1-channel intensity map.

	Attributes:
		channels (int):
            The number of output channels (or filters) for the `Conv2D` layer
            in the enhancenet. Default: `64`.
        kernel_size (Size2T):
            The kernel size for the `Conv2D` layer in the enhancenet.
            Default: `3`.
        use_batchnorm (bool):
            If `True`, use Batch Normalization layer between `Conv2D` and
            `Activation` layers. Default: `True`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels     : int    = 64,
		kernel_size  : Size2T = 3,
		use_batchnorm: bool   = True,
		*args, **kwargs
	):
		super().__init__()
		self.name 		   = "enhancenet"
		self.use_batchnorm = use_batchnorm
		
		channels 	= channels
		kernel_size = to_2tuple(kernel_size)
		
		self.relu    = nn.ReLU(inplace=True)
		self.conv0_1 = nn.Conv2d(
			4, channels, kernel_size, padding=1, padding_mode="replicate"
		)
		self.conv1_1 = nn.Conv2d(
			channels, channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.conv1_2 = nn.Conv2d(
			channels, channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.conv1_3 = nn.Conv2d(
			channels, channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.deconv1_1 = nn.Conv2d(
			channels * 2, channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv1_2 = nn.Conv2d(
			channels * 2, channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv1_3 = nn.Conv2d(
			channels * 2, channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.fusion = nn.Conv2d(
			channels * 3, channels, (1, 1), padding=1,
			padding_mode="replicate"
		)
		self.output = nn.Conv2d(channels, 1, (3, 3), padding=0)
		self.bn     = nn.BatchNorm2d(channels)
	
	# MARK: Forward Pass
	
	def forward(
		self, r_low: torch.Tensor, i_low: torch.Tensor
	) -> torch.Tensor:
		"""Forward pass.

		Args:
			r_low (torch.Tensor):
				The reflectance maps extracted from the low-light images.
			i_low (torch.Tensor):
				The illumination maps extracted from the low-light images.

		Returns:
			y_hat (torch.Tensor):
				The enhanced illumination map (a.k.a, i_delta).
		"""
		x     = torch.cat(tensors=(r_low, i_low), dim=1)
		conv0 = self.conv0_1(x)
		conv1 = self.relu(self.conv1_1(conv0))
		conv2 = self.relu(self.conv1_1(conv1))
		conv3 = self.relu(self.conv1_1(conv2))
		
		conv3_up = F.interpolate(
			input=conv3, size=(conv2.size()[2], conv2.size()[3])
		)
		deconv1 = self.relu(
			self.deconv1_1(torch.cat(tensors=(conv3_up, conv2), dim=1))
		)
		deconv1_up = F.interpolate(
			input=deconv1, size=(conv1.size()[2], conv1.size()[3])
		)
		deconv2 = self.relu(
			self.deconv1_2(torch.cat(tensors=(deconv1_up, conv1), dim=1))
		)
		deconv2_up = F.interpolate(
			input=deconv2, size=(conv0.size()[2], conv0.size()[3])
		)
		deconv3 = self.relu(
			self.deconv1_3(torch.cat(tensors=(deconv2_up, conv0), dim=1))
		)
		
		deconv1_rs = F.interpolate(
			input=deconv1, size=(r_low.size()[2], r_low.size()[3])
		)
		deconv2_rs = F.interpolate(
			input=deconv2, size=(r_low.size()[2], r_low.size()[3])
		)
		feats_all = torch.cat(tensors=(deconv1_rs, deconv2_rs, deconv3), dim=1)
		feats_fus = self.fusion(feats_all)
		
		if self.use_batchnorm:
			feats_fus = self.bn(feats_fus)
		y_hat = self.output(feats_fus)
		return y_hat


# MARK: - EnhanceUNet (Unet-based Variance)

# noinspection PyMethodOverriding
class EnhanceUNet(nn.Module):
	"""EnhanceUNet is a variation of the EnhanceNet that adopts the UNet
	architecture. EnhanceUNet breaks the RGB image into a 1-channel intensity
	map and a 3-channels reflectance map.

	Attributes:
		channels (int):
			The number of output channels (or filters) for the `Conv2D` layer
			in the enhancenet. Default: `64`.
		kernel_size (Size2T):
			The kernel size for the `Conv2D` layer in the enhancenet.
			Default: `3`.
		fuse_features (bool):
			If `True`, fuse features from all layers at the end.
			Default: `False`.
		use_batchnorm (bool):
			If `True`, use Batch Normalization layer between `Conv2D` and
			`Activation` layers. Default: `True`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels     : int 	  = 64,
		kernel_size  : Size2T = 3,
		fuse_features: bool   = False,
		use_batchnorm: bool   = True,
		*args, **kwargs
	):
		super().__init__()
		self.name 		   = "enhance_unet"
		self.use_batchnorm = use_batchnorm
		self.relu          = nn.ReLU()
		
		channels 	  = channels
		kernel_size   = to_2tuple(kernel_size)
		fuse_features = fuse_features
		
		# Downscale
		conv1_channels = self.channels
		self.conv1_1   = nn.Conv2d(
			4, conv1_channels, kernel_size, padding=1, padding_mode="replicate")
		self.conv1_2 = nn.Conv2d(
			conv1_channels, conv1_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		
		conv2_channels = conv1_channels * 2
		self.conv2_1 = nn.Conv2d(
			conv1_channels, conv2_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.conv2_2 = nn.Conv2d(
			conv2_channels, conv2_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		
		conv3_channels = conv2_channels * 2
		self.conv3_1 = nn.Conv2d(
			conv2_channels, conv3_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.conv3_2 = nn.Conv2d(
			conv3_channels, conv3_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		
		conv4_channels = conv3_channels * 2
		self.conv4_1 = nn.Conv2d(
			conv3_channels, conv4_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		self.conv4_2 = nn.Conv2d(
			conv4_channels, conv4_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		
		conv5_channels = conv4_channels * 2
		self.conv5_1 = nn.Conv2d(
			conv4_channels, conv5_channels, kernel_size, (2, 2), padding=1,
			padding_mode="replicate"
		)
		
		# Upscale
		self.deconv4_1 = nn.Conv2d(
			conv5_channels, conv4_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv4_2 = nn.Conv2d(
			conv4_channels * 2, conv4_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv4_3 = nn.Conv2d(
			conv4_channels, conv1_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		
		self.deconv3_1 = nn.Conv2d(
			conv4_channels, conv3_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv3_2 = nn.Conv2d(
			conv3_channels * 2, conv3_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv3_3 = nn.Conv2d(
			conv3_channels, conv1_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		
		self.deconv2_1 = nn.Conv2d(
			conv3_channels, conv2_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv2_2 = nn.Conv2d(
			conv2_channels * 2, conv2_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv2_3 = nn.Conv2d(
			conv2_channels, conv1_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		
		self.deconv1_1 = nn.Conv2d(
			conv2_channels, conv1_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		self.deconv1_2 = nn.Conv2d(
			conv1_channels * 2, conv1_channels, kernel_size, padding=1,
			padding_mode="replicate"
		)
		
		# Fusion and output
		self.fusion = nn.Conv2d(
			conv1_channels * 4, conv1_channels, (1, 1), padding=1,
			padding_mode="replicate"
		)
		self.output = nn.Conv2d(conv1_channels, 1, (3, 3), padding=0)
		self.bn 	= nn.BatchNorm2d(channels)
	
	# MARK: Forward Pass
	
	def forward(self, r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
		"""Forward pass.

		Args:
			r (torch.Tensor):
				The reflectance map.
			i (torch.Tensor):
				The illumination map.

		Returns:
			y_hat (torch.Tensor):
				The enhanced illumination map (a.k.a, i_delta).
		"""
		x = torch.cat(tensors=(r, i), dim=1)
		
		# Downsample path
		conv1_1 = self.conv1_1(x)
		conv1_2 = self.relu(self.conv1_2(conv1_1))
		
		conv2_1 = self.relu(self.conv2_1(conv1_2))
		conv2_2 = self.relu(self.conv2_2(conv2_1))
		
		conv3_1 = self.relu(self.conv3_1(conv2_2))
		conv3_2 = self.relu(self.conv3_2(conv3_1))
		
		conv4_1 = self.relu(self.conv4_1(conv3_2))
		conv4_2 = self.relu(self.conv4_2(conv4_1))
		
		conv5_1 = self.relu(self.conv5_1(conv4_2))
		
		# Upsample path
		deconv4_1    = self.relu(self.deconv4_1(conv5_1))
		deconv4_1_up = F.interpolate(
			input=deconv4_1, size=(conv4_2.size()[2], conv4_2.size()[3])
		)
		deconv4_2 	 = self.relu(
			self.deconv4_2(torch.cat(tensors=(deconv4_1_up, conv4_2), dim=1))
		)
		
		deconv3_1    = self.relu(self.deconv3_1(deconv4_2))
		deconv3_1_up = F.interpolate(
			input=deconv3_1, size=(conv3_2.size()[2], conv3_2.size()[3])
		)
		deconv3_2 	 = self.relu(
			self.deconv3_2(torch.cat(tensors=(deconv3_1_up, conv3_2), dim=1))
		)
		
		deconv2_1    = self.relu(self.deconv2_1(deconv3_2))
		deconv2_1_up = F.interpolate(
			input=deconv2_1, size=(conv2_2.size()[2], conv2_2.size()[3])
		)
		deconv2_2 	 = self.relu(
			self.deconv2_2(torch.cat(tensors=(deconv2_1_up, conv2_2), dim=1))
		)
		
		deconv1_1    = self.relu(self.deconv1_1(deconv2_2))
		deconv1_1_up = F.interpolate(
			input=deconv1_1, size=(conv1_2.size()[2], conv1_2.size()[3])
		)
		deconv1_2 	 = self.relu(
			self.deconv1_2(torch.cat(tensors=(deconv1_1_up, conv1_2), dim=1))
		)
		deconv1_2_rs = F.interpolate(
			input=deconv1_2, size=(r.size()[2], r.size()[3])
		)
		
		final_layer = deconv1_2_rs
		if self.fuse_features:
			deconv4_3    = self.relu(self.deconv4_3(deconv4_2))
			deconv4_3_rs = F.interpolate(
				input=deconv4_3, size=(r.size()[2], r.size()[3])
			)
			deconv3_3    = self.relu(self.deconv3_3(deconv3_2))
			deconv3_3_rs = F.interpolate(
				input=deconv3_3, size=(r.size()[2], r.size()[3])
			)
			deconv2_3    = self.relu(self.deconv2_3(deconv2_2))
			deconv2_3_rs = F.interpolate(
				input=deconv2_3, size=(r.size()[2], r.size()[3])
			)
			feats_all = torch.cat(
				tensors=(deconv4_3_rs, deconv3_3_rs, deconv2_3_rs,
						 deconv1_2_rs),
				dim=1
			)
			final_layer = self.fusion(feats_all)
		
		if self.use_batchnorm:
			final_layer = self.bn(final_layer)
		output = self.output(final_layer)
		y_hat  = F.interpolate(input=output, size=(r.size()[2], r.size()[3]))
		return y_hat
	
	
# MARK: - Phase

class Phase(Enum):
	"""The phases of the Retinex model."""

	# Train the DecomNet ONLY. Produce predictions, calculate losses and
	# metrics, update weights at the end of each epoch/step.
	DECOMNET   = "decomnet"
	# Train the EnhanceNet ONLY. Produce predictions, calculate losses and
	# metrics, update weights at the end of each epoch/step.
	ENHANCENET = "enhancenet"
	# Train the whole network. Produce predictions, calculate losses and
	# metrics, update weights at the end of each epoch/step.
	RETINEXNET = "retinexnet"

	TRAINING = "training"
	# Produce predictions, calculate losses and metrics, DO NOT update weights
	# at the end of each epoch/step.
	TESTING    = "testing"
	# Produce predictions ONLY.
	INFERENCE  = "inference"
	
	@staticmethod
	def values() -> list[str]:
		"""Return the list of all values.

		Returns:
			(list):
				The list of string.
		"""
		return [e.value for e in Phase]

	@staticmethod
	def keys():
		"""Return the list of all enum keys.

		Returns:
			(list):
				The list of enum keys.
		"""
		return [e for e in Phase]
	

# MARK: - RetinexNet

cfgs = {
	"A": {
		"decomnet":
			[DecomNet, dict(num_activation_layers=5, channels=64,
							kernel_size=3, use_batchnorm=True)],
		"enhancenet":
			[EnhanceNet, dict(channels=64, kernel_size=3, use_batchnorm=True)]
	},
	"B": {
		"decomnet":
			[DecomNet, dict(num_activation_layers=5, channels=64,
							kernel_size=3, use_batchnorm=True)],
		"enhancenet":
			[EnhanceUNet, dict(channels=64, kernel_size=3, fuse_features=False,
							   use_batchnorm=True)]
	},
}


# noinspection PyMethodOverriding,PyDictCreation
@ENHANCERS.register(name="retinexnet")
@MODELS.register(name="retinexnet")
class RetinexNet(BaseModel):
	"""Retinex-based models combine two submodels: DecomNet and EnhanceNet.
	RetinexNet is a multi-stage-training enhancer. We have to train the
	DecomNet first and then train the EnhanceNet.
	
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
			Default: `A`.

	Notes:
		- When training the DecomNet: epoch=75, using Adam(lr=0.00001) gives
		  best results.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		cfg        : Union[str, list, dict],
		phase      : Phase         		    = Phase.TRAINING,
		metrics	   : Optional[list[dict]]   = None,
		name       : Optional[str] 		    = "retinexnet",
		out_indexes: Indexes				= -1,
		pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			name=name, pretrained=pretrained,  out_indexes=out_indexes,
			*args, **kwargs
		)
		# NOTE: Get Hyperparameters
		if isinstance(cfg, str) and cfg in cfgs:
			cfg = cfgs[cfg]
		assert isinstance(cfg, dict)
		self.cfg = cfg
		
		decomnet,   decomnet_cfg   = cfg["decomnet"]
		enhancenet, enhancenet_cfg = cfg["enhancenet"]
		self.decomnet     = decomnet(**decomnet_cfg)
		self.enhancenet   = enhancenet(**enhancenet_cfg)
		self.phase 		  = phase
		self.decom_loss	  = DecomLoss()
		self.enhance_loss = EnhanceLoss()
		self.retinex_loss = RetinexLoss()
		self.metrics 	  = METRICS.build_from_dictlist(cfgs=metrics)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
	
	# MARK: Property

	# noinspection PyAttributeOutsideInit
	@BaseModel.phase.setter
	def phase(self, phase: Phase):
		"""Configure the model's running phase.

		Args:
			phase (Phase):
				The phase of the model.
		"""
		# assert phase in Phase, f"The model's `phase` must be one of the
		# following values: {Phase.keys()}"
		self._phase = phase
		
		if self._phase is Phase.DECOMNET:
			self.decomnet.train()
			self.enhancenet.eval()
		elif self._phase is Phase.ENHANCENET:
			self.decomnet.eval()
			self.enhancenet.train()
		elif self._phase is Phase.RETINEXNET:
			self.decomnet.train()
			self.enhancenet.train()
		elif self._phase in [Phase.TESTING, Phase.INFERENCE]:
			self.decomnet.eval()
			self.enhancenet.eval()
	
	@BaseModel.debug_image_filepath.getter
	def debug_image_filepath(self) -> str:
		"""Return the debug image filepath."""
		save_to_subdir = getattr(self.debug_cfg, "save_to_subdir", False)
		save_dir       = (self.debug_image_dir if save_to_subdir
						  else self.debug_dir)
		
		return os.path.join(
			save_dir, (f"{self.phase.value}_"
					   f"{(self.current_epoch + 1):03d}_"
					   f"{(self.epoch_step + 1):06}.jpg")
		)
	
	# MARK: Forward Pass
	
	def forward_train(
		self, x: torch.Tensor, y: torch.Tensor
	) -> ForwardXYOutput:
		"""Forward pass with both `x` and `y` are given.
		
		Args:
			x (torch.Tensor):
				The low-light images of shape [B, C, H, W].
			y (torch.Tensor, optional):
				The normal-light images (a.k.a ground truth) of shape
				[B, C, H, W].

		Returns:
			y_hat (Tensors):
				The final predictions consisting of (r_low, i_low, i_delta,
				enhanced image).
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		if self.phase is Phase.DECOMNET:
			r_low,  i_low   = self.decomnet(x=x)
			r_high, i_high  = self.decomnet(x=y)
			metrics 		= {}
			metrics["loss"] = self.decom_loss(x, y, r_low, r_high, i_low, i_high)
			if self.with_metrics:
				ms 		= {m.name: m(r_high, r_low) for m in self.metrics}
				metrics = metrics | ms  # NOTE: 3.9+ ONLY
			y_hat = (r_low, r_high, i_low, i_high)
			return y_hat, metrics
		
		elif self.phase is Phase.ENHANCENET:
			r_low,  i_low   = self.decomnet(x=x)
			i_delta         = self.enhancenet(r_low=r_low, i_low=i_low)
			i_delta_3 	    = torch.cat((i_delta, i_delta, i_delta), dim=1)
			y_hat 		    = r_low * i_delta_3
			metrics		    = {}
			metrics["loss"] = self.enhance_loss(
				y, r_low, i_delta, i_delta_3, y_hat
			)
			if self.with_metrics:
				ms 		= {m.name: m(y_hat, y) for m in self.metrics}
				metrics = metrics | ms  # NOTE: 3.9+ ONLY
			y_hat = (r_low, i_low, i_delta, y_hat)
			return y_hat, metrics
		
		else:
			r_low,  i_low   = self.decomnet(x=x)
			r_high, i_high  = self.decomnet(x=y)
			i_delta         = self.enhancenet(r_low=r_low, i_low=i_low)
			i_delta_3 	    = torch.cat((i_delta, i_delta, i_delta), dim=1)
			y_hat 		    = r_low * i_delta_3
			metrics		    = {}
			metrics["loss"] = self.retinex_loss(
				x, y, r_low, r_high, i_low, i_high, i_delta
			)
			if self.with_metrics:
				ms 		= {m.name: m(y_hat, y) for m in self.metrics}
				metrics = metrics | ms  # NOTE: 3.9+ ONLY
			y_hat = (r_low, i_low, i_delta, y_hat)
			return y_hat, metrics
		
	def forward_infer(self, x: torch.Tensor) -> Tensors:
		"""Forward pass with only `x` is given.

		Args:
			x (torch.Tensor):
				 The low-light images of shape [B, C, H, W].

		Returns:
			y_hat (Tensors):
				The final predictions consisting of (r_low, i_low, i_delta,
				enhanced image).
		"""
		r_low, i_low = self.decomnet(x=x)
		i_delta      = self.enhancenet(r_low=r_low, i_low=i_low)
		i_delta_3 	 = torch.cat((i_delta, i_delta, i_delta), dim=1)
		y_hat 		 = r_low * i_delta_3
		y_hat        = (r_low, i_low, i_delta, y_hat)
		return y_hat
 	
	# MARK: Visualization
	
	def show_results(
		self,
		x	 		 : Images,
		y	 		 : Optional[Images] = None,
		y_hat		 : Optional[Images] = None,
		filepath	 : Optional[str]    = None,
		image_quality: int              = 95,
		show         : bool             = False,
		show_max_n   : int              = 8,
		wait_time    : float            = 0.01,
		*args, **kwargs
	):
		"""Draw `result` over input image.

		Args:
			x (Images, Arrays):
				The low-light images.
			y (Images, Arrays, optional):
				The normal-light images.
			y_hat (Images, Arrays, optional):
				The predictions. When `phase=DECOMNET`, it is (r_low, r_high,
				i_low, i_high). Otherwise, it is (r_low, i_low, i_delta,
				enhanced image).
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
		# NOTE: Prepare images
		x       = to_4d_array(x)
		y       = to_4d_array(y)              if y     is not None else None
		y_hat   = self.prepare_results(y_hat) if y_hat is not None else None
		
		if self.phase is Phase.DECOMNET:
			(r_low, r_high, i_low, i_high) = y_hat
			results = dict(low=x, high=y, r_low=r_low, r_high=r_high,
						   i_low=i_low, i_high=i_high)
		else:
			(r_low, i_low, i_delta, enhance) = y_hat
			results = dict(low=x, r_low=r_low, i_low=i_low, i_delta=i_delta,
						   enhance=enhance, high=y)
		
		filepath = self.debug_image_filepath if filepath is None else filepath
		save_cfg = dict(filepath=filepath,
						pil_kwargs=dict(quality=image_quality))
		imshow_plt(images=results, scale=2, save_cfg=save_cfg, show=show,
				   show_max_n=show_max_n, wait_time=wait_time)
	
	def prepare_results(self, y_hat: Images) -> Arrays:
		"""Prepare results for visualization.

		Args:
			y_hat (Images, Arrays):
				The predictions. When `phase=DECOMNET`, it is (r_low, r_high,
				i_low, i_high). Otherwise, it is (r_low, i_low, i_delta,
				enhanced image).

		Returns:
			results (Arrays):
				The results for visualization.
		"""
		if self.phase is Phase.DECOMNET:
			(r_low, r_high, i_low, i_high) = y_hat
			i_low_3  = torch.cat(tensors=(i_low,  i_low,  i_low),  dim=1)
			i_high_3 = torch.cat(tensors=(i_high, i_high, i_high), dim=1)

			r_low    = to_4d_array(x=r_low)
			r_high   = to_4d_array(x=r_high)
			i_low_3  = to_4d_array(x=i_low_3)
			i_high_3 = to_4d_array(x=i_high_3)
			return r_low, r_high, i_low_3, i_high_3

		elif self.phase in [Phase.ENHANCENET, Phase.RETINEXNET, Phase.TESTING,
							Phase.INFERENCE]:
			(r_low, i_low, i_delta, enhance) = y_hat
			i_low_3   = torch.cat(tensors=(i_low,   i_low,   i_low),   dim=1)
			i_delta_3 = torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
			
			r_low     = to_4d_array(x=r_low)
			i_low_3   = to_4d_array(x=i_low_3)
			i_delta_3 = to_4d_array(x=i_delta_3)
			enhance   = to_4d_array(x=enhance)
			return r_low, i_low_3, i_delta_3, enhance
			
		return y_hat
	

# MARK: - RetinexUNet
@ENHANCERS.register(name="retinex_unet")
@MODELS.register(name="retinex_unet")
class RetinexUNet(RetinexNet):
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		phase     : Phase         		   = Phase.TRAINING,
		metrics	  : Optional[list[dict]]   = None,
		name      : Optional[str] 		   = "retinexnet",
		pretrained: Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="B", phase=phase, metrics=metrics, name=name,
			pretrained=pretrained, *args, **kwargs
		)
