#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MPRNet defined in the paper "Multi-Stage Progressive Image Restoration".
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Sequence
from typing import Union

import torch
from torch import nn

from torchkit.core.layer import CAB
from torchkit.core.layer import Downsample
from torchkit.core.layer import ORB
from torchkit.core.layer import SAM
from torchkit.core.layer import SkipUpsample
from torchkit.core.layer import Upsample
from torchkit.core.utils import Indexes
from torchkit.core.utils import Size2T
from torchkit.core.utils import Tensors
from torchkit.core.utils import to_2tuple
from torchkit.models.builder import ENHANCERS
from torchkit.models.builder import MODELS
from .e2e_enhancer import End2EndEnhancer
from ...core.utils import Arrays
from ...core.utils import ForwardXYOutput

logger = logging.getLogger()


# MARK: - UNetEncoder

class UNetEncoder(nn.Module):
	"""Modified UNet Encoder.
	
	Attributes:
		num_features (int):
		kernel_size (int, tuple[int]):
			The kernel size of the convolution layer.
		reduction (int):
			Reduction factor.
		bias (bool):
		act (nn.Module):
			The activation function of the convolution layer.
		scale_unetfeats (int):
		csff (bool):
			Should use "Cross Stage Feature Fusion"?
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		num_features   : int,
		kernel_size    : Size2T,
		reduction      : int,
		bias           : bool,
		act            : nn.Module,
		scale_unetfeats: int,
		csff           : bool,
	):
		super().__init__()
		self.num_features    = num_features
		self.kernel_size     = to_2tuple(kernel_size)
		self.reduction       = reduction
		self.bias            = bias
		self.act             = act
		self.scale_unetfeats = scale_unetfeats
		self.csff            = csff
		
		encoder_level1      = [
			CAB(
				channels    = self.num_features,
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		encoder_level2      = [
			CAB(
				channels    = self.num_features + self.scale_unetfeats,
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		encoder_level3      = [
			CAB(
				channels    = self.num_features + (self.scale_unetfeats * 2),
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		self.encoder_level1 = nn.Sequential(*encoder_level1)
		self.encoder_level2 = nn.Sequential(*encoder_level2)
		self.encoder_level3 = nn.Sequential(*encoder_level3)

		self.down12 = Downsample(
			in_channels  = self.num_features,
			scale_factor = self.scale_unetfeats
		)
		self.down23 = Downsample(
			in_channels  = self.num_features + self.scale_unetfeats,
			scale_factor = self.scale_unetfeats
		)

		# NOTE: Cross Stage Feature Fusion (CSFF)
		if self.csff:
			self.csff_enc1 = nn.Conv2d(
				in_channels  = num_features,
				out_channels = num_features,
				kernel_size  = (1, 1),
				bias         = self.bias
			)
			self.csff_enc2 = nn.Conv2d(
				in_channels  = self.num_features + self.scale_unetfeats,
				out_channels = self.num_features + self.scale_unetfeats,
				kernel_size  = (1, 1),
				bias         = self.bias
			)
			self.csff_enc3 = nn.Conv2d(
				in_channels  = self.num_features + (self.scale_unetfeats * 2),
				out_channels = self.num_features + (self.scale_unetfeats * 2),
				kernel_size  = (1, 1),
				bias         = self.bias
			)
			self.csff_dec1 = nn.Conv2d(
				in_channels  = self.num_features,
				out_channels = self.num_features,
				kernel_size  = (1, 1),
				bias         = self.bias
			)
			self.csff_dec2 = nn.Conv2d(
				in_channels  = self.num_features + self.scale_unetfeats,
				out_channels = self.num_features + self.scale_unetfeats,
				kernel_size  = (1, 1),
				bias         = self.bias
			)
			self.csff_dec3 = nn.Conv2d(
				in_channels  = self.num_features + (self.scale_unetfeats * 2),
				out_channels = self.num_features + (self.scale_unetfeats * 2),
				kernel_size  = (1, 1),
				bias         = self.bias
			)
	
	# MARK: Forward Pass
	
	def forward(
		self,
		x           : torch.Tensor,
		encoder_outs: Optional[Tensors] = None,
		decoder_outs: Optional[Tensors] = None
	) -> Sequence[torch.Tensor]:
		"""Forward pass.

		Args:
			x (torch.Tensor):
				The input images or images' patches.
			encoder_outs (Tensors, optional):
				UNet encoder's features from previous stage.
			decoder_outs (Tensors, optional):
				UNet decoder's features from previous stage.
				
		Returns:
			y_hat (Sequence[torch.Tensor]):
				The output list of tensors.
		"""
		enc1 = self.encoder_level1(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc1 = (enc1 +
					self.csff_enc1(encoder_outs[0]) +
					self.csff_dec1(decoder_outs[0]))

		x    = self.down12(enc1)
		enc2 = self.encoder_level2(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc2 = (enc2 +
					self.csff_enc2(encoder_outs[1]) +
					self.csff_dec2(decoder_outs[1]))

		x    = self.down23(enc2)
		enc3 = self.encoder_level3(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc3 = (enc3 +
					self.csff_enc3(encoder_outs[2]) +
					self.csff_dec3(decoder_outs[2]))
		
		y_hat = [enc1, enc2, enc3]
		return y_hat


# MARK: - UNetDecoder

class UNetDecoder(nn.Module):
	"""Modified UNet Decoder.
	
	Attributes:
		num_features (int):
		kernel_size (int, tuple[int]):
			The kernel size of the convolution layer.
		reduction (int):
			Reduction factor.
		bias (bool):
		act (nn.Module):
			The activation function of the convolution layer.
		scale_unetfeats (int):
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		num_features   : int,
		kernel_size    : Size2T,
		reduction      : int,
		bias           : bool,
		act            : nn.Module,
		scale_unetfeats: int,
	):
		super().__init__()
		self.num_features    = num_features
		self.kernel_size     = to_2tuple(kernel_size)
		self.reduction       = reduction
		self.bias            = bias
		self.act             = act
		self.scale_unetfeats = scale_unetfeats
		
		decoder_level1      = [
			CAB(
				channels    = self.num_features,
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		decoder_level2      = [
			CAB(
				channels    = self.num_features + self.scale_unetfeats,
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		decoder_level3      = [
			CAB(
				channels    = self.num_features + (self.scale_unetfeats * 2),
				kernel_size = self.kernel_size,
				reduction   = self.reduction,
				bias        = self.bias,
				act         = self.act
			) for _ in range(2)
		]
		self.decoder_level1 = nn.Sequential(*decoder_level1)
		self.decoder_level2 = nn.Sequential(*decoder_level2)
		self.decoder_level3 = nn.Sequential(*decoder_level3)

		self.skip_attn1 = CAB(
			channels    = self.num_features,
			kernel_size = self.kernel_size,
			reduction   = self.reduction,
			bias        = self.bias,
			act         = self.act
		)
		self.skip_attn2 = CAB(
			channels    = self.num_features + self.scale_unetfeats,
			kernel_size = self.kernel_size,
			reduction   = self.reduction,
			bias        = self.bias,
			act         = self.act
		)
		self.up21 = SkipUpsample(
			in_channels  = self.num_features,
			scale_factor = self.scale_unetfeats
		)
		self.up32 = SkipUpsample(
			in_channels  = self.num_features + self.scale_unetfeats,
			scale_factor = self.scale_unetfeats
		)
	
	# MARK: Forward Pass
	
	def forward(self, x: Tensors) -> Tensors:
		"""Forward pass.

		Args:
			x (Tensors):
				The list of output tensors from the encoder.
				
		Returns:
			y_hat (Tensors):
				The output list of tensors.
		"""
		enc1, enc2, enc3 = x
		dec3 = self.decoder_level3(enc3)

		x     = self.up32(dec3, self.skip_attn2(enc2))
		dec2  = self.decoder_level2(x)

		x     = self.up21(dec2, self.skip_attn1(enc1))
		dec1  = self.decoder_level1(x)
		
		y_hat = [dec1, dec2, dec3]
		return y_hat


# MARK: - ORSNet

class ORSNet(nn.Module):
	"""Original-resolution network.
	
	Attributes:
		num_features (int):
		scale_orsnetfeats (int):
		kernel_size (int, tuple[int]):
			The kernel size of the convolution layer.
		reduction (int):
			Reduction factor.
		bias (bool):
		act (nn.Module):
			The activation function of the convolution layer.
		scale_unetfeats (int):
		num_cab (int):
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		num_features     : int,
		scale_orsnetfeats: int,
		kernel_size      : Size2T,
		reduction        : int,
		bias             : bool,
		act              : nn.Module,
		scale_unetfeats  : int,
		num_cab          : int,
	):
		super().__init__()
		self.num_features      = num_features
		self.scale_orsnetfeats = scale_orsnetfeats
		self.kernel_size       = to_2tuple(kernel_size)
		self.reduction         = reduction
		self.bias              = bias
		self.act               = act
		self.scale_unetfeats   = scale_unetfeats
		self.num_cab           = num_cab
		
		self.orb1 = ORB(
			channels    = self.num_features + self.scale_orsnetfeats,
			kernel_size = self.kernel_size,
			reduction   = self.reduction,
			bias        = self.bias,
			act         = self.act,
			num_cab     = self.num_cab
		)
		self.orb2 = ORB(
			channels    = self.num_features + self.scale_orsnetfeats,
			kernel_size = self.kernel_size,
			reduction   = self.reduction,
			bias        = self.bias,
			act         = self.act,
			num_cab     = self.num_cab
		)
		self.orb3 = ORB(
			channels    = self.num_features + self.scale_orsnetfeats,
			kernel_size = self.kernel_size,
			reduction   = self.reduction,
			bias        = self.bias,
			act         = self.act,
			num_cab     = self.num_cab
		)
		
		self.up_enc1 = Upsample(in_channels=self.num_features,
								scale_factor=self.scale_unetfeats)
		self.up_dec1 = Upsample(in_channels=self.num_features,
								scale_factor=self.scale_unetfeats)
		
		self.up_enc2 = nn.Sequential(
			Upsample(in_channels=self.num_features + self.scale_unetfeats,
					 scale_factor=self.scale_unetfeats),
			Upsample(in_channels=self.num_features,
					 scale_factor=self.scale_unetfeats)
		)
		self.up_dec2 = nn.Sequential(
			Upsample(in_channels=self.num_features + self.scale_unetfeats,
					 scale_factor=self.scale_unetfeats),
			Upsample(in_channels=self.num_features,
					 scale_factor=self.scale_unetfeats)
		)

		self.conv_enc1 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)
		self.conv_enc2 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)
		self.conv_enc3 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)

		self.conv_dec1 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)
		self.conv_dec2 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)
		self.conv_dec3 = nn.Conv2d(
			in_channels  = self.num_features,
			out_channels = self.num_features + self.scale_orsnetfeats,
			kernel_size  = (1, 1),
			bias         = self.bias
		)
	
	# MARK: Forward Pass
	
	def forward(
		self, x: torch.Tensor, encoder_outs: Tensors, decoder_outs: Tensors
	) -> torch.Tensor:
		"""Forward pass.

		Args:
			x (torch.Tensor):
				The input images or images' patches.
			encoder_outs (Tensors, optional):
				UNet encoder's features from previous stage.
			decoder_outs (Tensors, optional):
				UNet decoder's features from previous stage.
				
		Returns:
			y_hat (torch.Tensor):
				The output tensor.
		"""
		x = self.orb1(x)
		x = (x +
			 self.conv_enc1(encoder_outs[0]) +
			 self.conv_dec1(decoder_outs[0]))

		x = self.orb2(x)
		x = (x +
			 self.conv_enc2(self.up_enc1(encoder_outs[1])) +
			 self.conv_dec2(self.up_dec1(decoder_outs[1])))

		x = self.orb3(x)
		x = (x +
			 self.conv_enc3(self.up_enc2(encoder_outs[2])) +
			 self.conv_dec3(self.up_dec2(decoder_outs[2])))

		y_hat = x
		return y_hat


# MARK: - MPRNet

cfgs = {
	"A": dict(in_channels=3, out_channels=3, kernel_size=3, num_features=80,
			  scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, reduction=4,
			  bias=False),  # De-noising
	"B": dict(in_channels=3, out_channels=3, kernel_size=3, num_features=40,
			  scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, reduction=4,
			  bias=False),  # De-rain / De-snow
    "C": dict(in_channels=3, out_channels=3, kernel_size=3, num_features=96,
			  scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, reduction=4,
			  bias=False),  # De-blur
}


# noinspection PyDefaultArgument,PyMethodOverriding
@ENHANCERS.register(name="mprnet")
@MODELS.register(name="mprnet")
class MPRNet(End2EndEnhancer):
	"""MPRNet consists of three stages.
	
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
			Default: `dict(channels=8, kernel_size=5, num_blocks=10)`.
	
	Args:
		name (str, optional):
			Name of the backbone. Default: `mprnet`.
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
        name       : Optional[str]          = "mprnet",
		out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			name=name, pretrained=pretrained, out_indexes=out_indexes,
			*args, **kwargs
		)
		# NOTE: Get Hyperparameters
		if isinstance(cfg, str) and cfg in cfgs:
			cfg = cfgs[cfg]
		assert isinstance(cfg, dict)
		self.cfg = cfg
		
		in_channels       = cfg["in_channels"]
		out_channels      = cfg["out_channels"]
		num_features      = cfg["num_features"]
		scale_unetfeats   = cfg["scale_unetfeats"]
		scale_orsnetfeats = cfg["scale_orsnetfeats"]
		num_cab           = cfg["num_cab"]
		kernel_size       = to_2tuple(cfg["kernel_size"])
		padding           = kernel_size[0] // 2
		reduction         = cfg["reduction"]
		bias              = cfg["bias"]
		act          	  = nn.PReLU()
		
		# NOTE: Features
		self.shallow_feat1 = nn.Sequential(
			nn.Conv2d(in_channels, num_features, kernel_size, stride=(1, 1),
					  padding=padding, bias=bias),
			CAB(num_features, kernel_size, reduction, bias, act)
		)
		self.shallow_feat2 = nn.Sequential(
			nn.Conv2d(in_channels, num_features, kernel_size, padding=padding,
					  bias=bias),
			CAB(num_features, kernel_size, reduction, bias, act)
		)
		self.shallow_feat3 = nn.Sequential(
			nn.Conv2d(in_channels, num_features, kernel_size, padding=padding,
					  bias=bias),
			CAB(num_features, kernel_size, reduction, bias, act)
		)
		
		# NOTE: Neck (Cross Stage Feature Fusion (CSFF))
		self.stage1_encoder = UNetEncoder(
			num_features, kernel_size, reduction, bias, act, scale_unetfeats,
			csff=False
		)
		self.stage1_decoder = UNetDecoder(
			num_features, kernel_size, reduction, bias, act, scale_unetfeats,
		)

		self.stage2_encoder = UNetEncoder(
			num_features, kernel_size, reduction, bias, act, scale_unetfeats,
			csff=True
		)
		self.stage2_decoder = UNetDecoder(
			num_features, kernel_size, reduction, bias, act, scale_unetfeats,
		)

		self.stage3_orsnet = ORSNet(
			num_features, scale_orsnetfeats, kernel_size, reduction, bias, act,
			scale_unetfeats, num_cab
		)

		self.concat12 = nn.Conv2d(
			num_features * 2, num_features, kernel_size, padding=padding,
			bias=bias
		)
		self.concat23 = nn.Conv2d(
			num_features * 2, num_features + scale_orsnetfeats, kernel_size,
			padding=padding, bias=bias
		)
		
		# NOTE: Head
		self.sam12 = SAM(channels=num_features, kernel_size=1, bias=bias)
		self.sam23 = SAM(channels=num_features, kernel_size=1, bias=bias)
		
		self.tail = nn.Conv2d(
			num_features + scale_orsnetfeats, out_channels, kernel_size,
			padding=padding, bias=bias
		)
		
		# NOTE: Load Pretrained
		if self.pretrained:
			self.load_pretrained()
		
	# MARK: Forward Pass
	
	def forward_train(
		self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs
	) -> ForwardXYOutput:
		"""Forward pass during training with both `x` and `y` are given.

		Args:
			x (torch.Tensor):
				The low-quality images of shape [B, C, H, W].
			y (torch.Tensor, optional):
				The high-quality images (a.k.a ground truth) of shape
				[B, C, H, W].

		Returns:
			y_hat (Tensors):
				The final predictions consisting of the enhanced image.
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		y_hat   = self.forward_infer(x=x)
		metrics = {}
		if self.with_loss:
			metrics["loss"] = self.loss(y_hat, y)
		
		y_hat = y_hat[0]
		if self.with_metrics:
			ms		= {m.name: m(y_hat, y) for m in self.metrics}
			metrics = metrics | ms  # NOTE: 3.9+ ONLY
		metrics = metrics if len(metrics) else None
		return y_hat, metrics
	
	def forward_infer(self, x: torch.Tensor) -> Tensors:
		"""Forward pass.

		Args:
			x (torch.Tensor):
				The input images.

		Returns:
			y_hat (Tensors):
				The list of 3 tensors at three stages.
		"""
		# NOTE: Original-resolution Image for Stage 3
		x3_img = x
		h      = x3_img.size()[2]
		w      = x3_img.size()[3]

		# NOTE: Multi-Patch Hierarchy: Split Image into four non-overlapping
		# patches
		# Two Patches for Stage 2
		x2_top_img  = x3_img[:, :, 0 : int(h / 2), :]
		x2_bot_img  = x3_img[:, :, int(h / 2) : h, :]

		# Four Patches for Stage 1
		x1_ltop_img = x2_top_img[:, :, :, 0 : int(w / 2)]
		x1_rtop_img = x2_top_img[:, :, :, int(w / 2) : w]
		x1_lbot_img = x2_bot_img[:, :, :, 0 : int(w / 2)]
		x1_rbot_img = x2_bot_img[:, :, :, int(w / 2) : w]

		##-------------------------------------------
		##-------------- Stage 1 --------------------
		##-------------------------------------------
		## Compute Shallow Features
		x1_ltop    = self.shallow_feat1(x1_ltop_img)
		x1_rtop    = self.shallow_feat1(x1_rtop_img)
		x1_lbot    = self.shallow_feat1(x1_lbot_img)
		x1_rbot    = self.shallow_feat1(x1_rbot_img)
		
		## Process features of all 4 patches with Encoder of Stage 1
		feat1_ltop = self.stage1_encoder(x1_ltop)
		feat1_rtop = self.stage1_encoder(x1_rtop)
		feat1_lbot = self.stage1_encoder(x1_lbot)
		feat1_rbot = self.stage1_encoder(x1_rbot)
		
		## Concat deep features
		feat1_top = [torch.cat((k, v), 3) for k, v in
					 zip(feat1_ltop, feat1_rtop)]
		feat1_bot = [torch.cat((k, v), 3) for k, v in
					 zip(feat1_lbot, feat1_rbot)]
		
		## Pass features through Decoder of Stage 1
		res1_top = self.stage1_decoder(feat1_top)
		res1_bot = self.stage1_decoder(feat1_bot)
		
		## Apply Supervised Attention Module (SAM)
		x2_top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2_top_img)
		x2_bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2_bot_img)

		## Output image at Stage 1
		stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
		
		##-------------------------------------------
		##-------------- Stage 2 --------------------
		##-------------------------------------------
		## Compute Shallow Features
		x2_top = self.shallow_feat2(x2_top_img)
		x2_bot = self.shallow_feat2(x2_bot_img)

		## Concatenate SAM features of Stage 1 with shallow features of Stage 2
		x2_top_cat = self.concat12(torch.cat([x2_top, x2_top_samfeats], 1))
		x2_bot_cat = self.concat12(torch.cat([x2_bot, x2_bot_samfeats], 1))

		## Process features of both patches with Encoder of Stage 2
		feat2_top  = self.stage2_encoder(x2_top_cat, feat1_top, res1_top)
		feat2_bot  = self.stage2_encoder(x2_bot_cat, feat1_bot, res1_bot)

		## Concat deep features
		feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

		## Pass features through Decoder of Stage 2
		res2 = self.stage2_decoder(feat2)

		## Apply SAM
		x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

		##-------------------------------------------
		##-------------- Stage 3 --------------------
		##-------------------------------------------
		## Compute Shallow Features
		x3 = self.shallow_feat3(x3_img)

		## Concatenate SAM features of Stage 2 with shallow features of Stage 3
		x3_cat     = self.concat23(torch.cat([x3, x3_samfeats], 1))
		x3_cat     = self.stage3_orsnet(x3_cat, feat2, res2)
		stage3_img = self.tail(x3_cat)

		return [stage3_img + x3_img, stage2_img, stage1_img]
	
	# MARK: Training
	
	def on_fit_start(self):
		"""Called at the very beginning of fit."""
		super().on_fit_start()
		if self.shape:
			h, w, c = self.shape
			assert h == 256 and w == 256, \
				(f"MBLLEN model requires image's shape to be [256, 256, :]. "
				 f"Got {self.shape}.")
	
	# MARK: Utils

	def prepare_results(
		self,
		y_hat: Union[Tensors, Arrays],
		x    : Optional[Union[Tensors, Arrays]] = None,
		y    : Optional[Union[Tensors, Arrays]] = None,
		*args, **kwargs
	) -> Arrays:
		"""Prepare results for visualization.

		Args:
			y_hat (Tensors, Arrays):
				The predictions.
			x (Tensors, Arrays, optional):
				The input images. Default: `None`.
			y (Tensors, Arrays, optional):
				The ground truth. Default: `None`.

		Returns:
			results (Arrays):
				The results for visualization.
		"""
		if isinstance(y_hat, (list, tuple)):
			return y_hat[0]
		else:
			return y_hat


@ENHANCERS.register(name="mprnet_blur")
@MODELS.register(name="mprnet_blur")
class MPRNetSnow(MPRNet):
	"""MPRNet consists of three stages."""
	
	models_zoo = {}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes                = -1,
        name       : Optional[str]          = "mprnet_blur",
        pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="A", out_indexes=out_indexes, name=name, pretrained=pretrained,
			*args, **kwargs
		)


@ENHANCERS.register(name="mprnet_rain")
@MODELS.register(name="mprnet_rain")
class MPRNetRain(MPRNet):
	"""MPRNet consists of three stages."""
	
	models_zoo = {}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes                = -1,
        name       : Optional[str]          = "mprnet_rain",
        pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="B", out_indexes=out_indexes, name=name, pretrained=pretrained,
			*args, **kwargs
		)


@ENHANCERS.register(name="mprnet_snow")
@MODELS.register(name="mprnet_snow")
class MPRNetSnow(MPRNet):
	"""MPRNet consists of three stages."""
	
	models_zoo = {}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		out_indexes: Indexes                = -1,
        name       : Optional[str]          = "mprnet_snow",
        pretrained : Union[bool, str, dict] = False,
		*args, **kwargs
	):
		super().__init__(
			cfg="B", out_indexes=out_indexes, name=name, pretrained=pretrained,
			*args, **kwargs
		)
