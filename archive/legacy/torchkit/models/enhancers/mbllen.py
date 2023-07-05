#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MBLLEN: Multi-Branch Low-Light Enhancement Network. The key idea is to
extract rich features up to different levels, so that we can apply enhancement
via multiple subnets and finally produce the output image via multi-branch
fusion. In this manner, image quality is improved from different aspects.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import torch
from torch import nn

from torchkit.core.layer import ConvReLU
from torchkit.core.utils import Indexes
from torchkit.core.utils import Size2T
from torchkit.core.utils import Tensors
from torchkit.core.utils import to_2tuple
from torchkit.models.builder import BACKBONES
from torchkit.models.builder import ENHANCERS
from torchkit.models.builder import MODELS
from .e2e_enhancer import End2EndEnhancer

logger = logging.getLogger()


# MARK: - EM

class EM(nn.Module):
    """The Enhancement regression (EM) has a symmetric structure to first apply
    convolutions and then deconvolutions.

    Attributes:
        channels (int:
            The number of channels for `Conv2D` layers used in each EM block.
            Default: `8`.
        kernel_size (Size2T):
            The kernel size for `Conv2D` layers used in each EM block.
            Default: `5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, channels: int = 8, kernel_size: Size2T = 5):
        super().__init__()
        self.channels    = channels
        self.kernel_size = to_2tuple(kernel_size)
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(32, self.channels, kernel_size=(3, 3), padding=1,
                      padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels * 2, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 4, self.kernel_size),
            nn.ReLU()
        )
        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(self.channels * 4, self.channels * 2,
                               self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels * 2, self.channels,
                               self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, 3, self.kernel_size),
            nn.ReLU(),
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.deconvolutions(x)
        return x
    

# MARK: - MBLLEN

cfgs = {
    "A": dict(channels=8, kernel_size=5, num_blocks=10)
}


# noinspection PyDefaultArgument,PyMethodOverriding
@BACKBONES.register(name="mbllen")
@ENHANCERS.register(name="mbllen")
@MODELS.register(name="mbllen")
class MBLLEN(End2EndEnhancer):
    """MBLLEN consists of three modules: the feature extraction regression
    (FEM), the enhancement regression (EM) and the fusion regression (FM).
    
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
			Name of the backbone. Default: `mbllen`.
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
		
    Notes:
        - MBLLEN model requires input shape to be [:, 256, 256].
        - Optimizer should be: dict(name="adam", lr=0.0001)
    """

    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        cfg        : Union[str, list, dict] = "A",
        name       : Optional[str]          = "mbllen",
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
        
        channels    = cfg["channels"]
        kernel_size = to_2tuple(cfg["kernel_size"])
        num_blocks  = cfg["num_blocks"]

        # NOTE: Features
        fem  = [nn.Conv2d(3, 32, (3, 3), padding=1, padding_mode="replicate")]
        fem += [ConvReLU(32, 32, (3, 3), padding=1, padding_mode="replicate")
                for _ in range(num_blocks - 1)]
        self.backbone = nn.ModuleList(fem)
    
        # NOTE: Neck
        em = [EM(channels, kernel_size) for _ in range(num_blocks)]
        self.neck = nn.ModuleList(em)
        
        # NOTE: Head
        fm = nn.Conv2d(30, 3, (1, 1), padding=0, padding_mode="replicate")
        self.head = fm
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
            
    # MARK: Forward Pass

    def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        fem_feat  = x
        em_concat = None
        for idx, (fem_i, em_i) in enumerate(zip(self.fem, self.em)):
            fem_feat  = fem_i(fem_feat)
            em_feat   = em_i(fem_feat)
            em_concat = (em_feat if (em_concat is None)
                         else torch.cat(tensors=(em_concat, em_feat), dim=1))
        y_hat = self.fm(em_concat)
        """
        em_concat = self.forward_features(x, out_indexes=-1)
        y_hat     = self.head(em_concat)
        return y_hat

    def forward_features(
        self, x: torch.Tensor, out_indexes: Optional[Indexes] = -1
    ) -> Tensors:
        """Forward pass for features extraction.
        	- If `out_indexes` is None, skip.
        	- If `out_indexes` is a tuple or list, return an array of features.
        	- If `out_indexes` is int, return only the feature from that
        	  layer's index.
        	- If `out_indexes` is -1, return the last layer's output.
        """
        out_indexes = self.out_indexes if out_indexes is None else out_indexes
       
        fem_feat    = x
        em_concat   = None
        y_hat       = []
        for idx, (fem_i, em_i) in enumerate(zip(self.backbone, self.neck)):
            fem_feat  = fem_i(fem_feat)
            em_feat   = em_i(fem_feat)
            em_concat = (em_feat if (em_concat is None)
                         else torch.cat(tensors=(em_concat, em_feat), dim=1))
            
            if isinstance(out_indexes, (tuple, list)) and idx in out_indexes:
                y_hat.append(x)
            elif isinstance(out_indexes, int) and idx == out_indexes:
                return em_concat
            else:
                y_hat = em_concat
        return y_hat

    # MARK: Training
    
    def on_fit_start(self):
        """Called at the very beginning of fit."""
        super().on_fit_start()
        if self.shape:
            h, w, c = self.shape
            assert h == 256 and w == 256, \
                (f"MBLLEN model requires image's shape to be [256, 256, :]. "
                 f"Got {self.shape}.")
