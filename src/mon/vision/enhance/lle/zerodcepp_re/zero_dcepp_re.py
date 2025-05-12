#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep
Curve Estimation," IEEE TPAMI 2022.

References:
    - https://github.com/Li-Chongyi/Zero-DCE_extension
"""

__all__ = [
    "ZeroDCEpp_RE",
]

from typing import Literal

import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import functional as F
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Loss -----
class Loss(nn.Loss):
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tva_weight    : float = 1600.0,
        reduction     : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tva_weight = tva_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tva = nn.TotalVariationLoss(reduction=reduction)
    
    def forward(
        self,
        input  : torch.Tensor,
        adjust : torch.Tensor,
        enhance: torch.Tensor,
        **_
    ) -> torch.Tensor:
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tva = self.loss_tva(input=adjust)
        loss     = (
              self.spa_weight * loss_spa
            + self.exp_weight * loss_exp
            + self.col_weight * loss_col
            + self.tva_weight * loss_tva
        )
        return loss


# ----- Model -----
@MODELS.register(name="zerodce++_re", arch="zerodce++")
class ZeroDCEpp_RE(base.ImageEnhancementModel):
    """Zero-DCE++ model for low-light image enhancement.
    
    Args:
        in_channels: The first layer's input channel. Default is ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent layers.
            Default is ``32``.
        num_iters: The number of convolutional layers in the model. Default is ``8``.
        scale_factor: Downsampling/upsampling ratio. Default is ``1.0``.
        
    References:
        - https://github.com/Li-Chongyi/Zero-DCE_extension
    """
    
    arch     : str          = "zerodce++"
    name     : str          = "zerodce++_re"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_REFERENCE]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}

    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        
        # Network
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        self.e_conv1  = nn.DSConv2d(in_channels,      num_channels, 3, 1, 1)
        self.e_conv2  = nn.DSConv2d(num_channels,     num_channels, 3, 1, 1)
        self.e_conv3  = nn.DSConv2d(num_channels,     num_channels, 3, 1, 1)
        self.e_conv4  = nn.DSConv2d(num_channels,     num_channels, 3, 1, 1)
        self.e_conv5  = nn.DSConv2d(num_channels * 2, num_channels, 3, 1, 1)
        self.e_conv6  = nn.DSConv2d(num_channels * 2, num_channels, 3, 1, 1)
        self.e_conv7  = nn.DSConv2d(num_channels * 2, 3, 3, 1, 1)
        
        # Loss
        self.loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        image    = datapoint["image"]
        enhanced = outputs["enhanced"]
        adjust   = outputs["adjust"]
        loss     = self.loss(image, adjust, enhanced)
        
        return outputs | {
            "loss": loss,
        }
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        # Input
        x      = datapoint["image"]
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        # Process
        x1  =  self.relu(self.e_conv1(x_down))
        x2  =  self.relu(self.e_conv2(x1))
        x3  =  self.relu(self.e_conv3(x2))
        x4  =  self.relu(self.e_conv4(x3))
        x5  =  self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  =  self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        
        # Enhance
        y = x
        for i in range(0, self.num_iters):
            y = y + x_r * (torch.pow(y, 2) - y)
        
        return {
            "adjust"  : x_r,
            "enhanced": y,
        }
