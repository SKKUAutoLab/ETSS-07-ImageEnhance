#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "HINet: Half Instance Normalization Network for Image
Restoration," CVPRW 2021.

References:
    - https://github.com/megvii-model/HINet
"""

__all__ = [
    "HINet_RE",
]

from typing import Any

import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Module -----
class UNetConvBlock(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool = False,
        use_hin     : bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.downsample = downsample
        self.identity   = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.use_csff   = use_csff

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if use_hin:
            self.norm = nn.InstanceNorm2d(out_channels // 2, affine=True)
        self.use_hin = use_hin

        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1, bias=False)

    def forward(
        self,
        input: torch.Tensor,
        enc  : torch.Tensor = None,
        dec  : torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = input
        y = self.conv_1(x)
        if self.use_hin:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y = torch.cat([self.norm(y1), y2], dim=1)
        y  = self.relu_1(y)
        y  = self.relu_2(self.conv_2(y))
        y += self.identity(x)
        if enc and dec:
            assert self.use_csff
            y = y + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            y_down = self.downsample(y)
            return y_down, y
        else:
            return y


class UNetUpBlock(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.up = nn.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv_block = UNetConvBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            downsample   = False,
            relu_slope   = relu_slope,
        )

    def forward(self, input: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.up(x)
        y = torch.cat([y, bridge], 1)
        y = self.conv_block(y)
        return y


class SupervisedAttentionModule(nn.Module):
    
    def __init__(
        self,
        channels    : int,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
    ):
        super().__init__()
        padding    = kernel_size[0] // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2 = nn.Conv2d(
            in_channels  = channels,
            out_channels = 3,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3 = nn.Conv2d(
            in_channels  = 3,
            out_channels = channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, x: torch.Tensor, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            x: The first tensor is the output from the previous layer.
            x_img: The second tensor is the current step input.
            
        Returns:
            Supervised attention features.
            Output feature for the next layer.
        """
        x1  = self.conv1(x)
        img = self.conv2(x) + x_img
        x2  = torch.sigmoid(self.conv3(img))
        x1  = x1 * x2
        x1  = x1 + x
        return x1, img


# ----- Model -----
@MODELS.register(name="hinet_re", arch="hinet")
class HINet_RE(base.ImageEnhancementModel):
    """Half-Instance Normalization Network.
    
    Args:
        in_channels: The first layer's input channel. Default is ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default is ``64``.
        depth: The depth of the network. Default is ``5``.
        relu_slope: The slope of the ReLU activation. Default is ``0.2``.
        in_pos_left: The layer index to begin applying the Instance Normalization.
            Default is ``0``.
        in_pos_right: The layer index to end applying the Instance Normalization.
            Default is ``4``.
        
    References:
        - https://github.com/megvii-model/HINet
    """
    
    arch     : str          = "hinet"
    name     : str          = "hinet_re"
    tasks    : list[Task]   = [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW, Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        num_channels: int   = 64,
        depth       : int   = 5,
        relu_slope  : float = 0.2,
        in_pos_left : int   = 0,
        in_pos_right: int   = 4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Network
        self.down_path_1  = nn.ModuleList()
        self.down_path_2  = nn.ModuleList()
        self.conv_01      = nn.Conv2d(3, num_channels, 3, 1, 1)
        self.conv_02      = nn.Conv2d(3, num_channels, 3, 1, 1)
        prev_channels     = num_channels
        for i in range(depth):  # 0, 1, 2, 3, 4
            use_hin    = True if in_pos_left <= i <= in_pos_right else False
            downsample = True if (i + 1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2 ** i) * num_channels, downsample, relu_slope, use_hin=use_hin))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2 ** i) * num_channels, downsample, relu_slope, use_csff=downsample, use_hin=use_hin))
            prev_channels = (2 ** i) * num_channels
        self.up_path_1   = nn.ModuleList()
        self.up_path_2   = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2 ** i) * num_channels, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2 ** i) * num_channels, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2 ** i) * num_channels, (2 ** i) * num_channels, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2 ** i) * num_channels, (2 ** i) * num_channels, 3, 1, 1))
            prev_channels = (2 ** i) * num_channels
        self.sam12 = SupervisedAttentionModule(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last  = nn.Conv2d(prev_channels, 3, 3, 1, 1, bias=True)
        
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
        gain      = torch.nn.init.calculate_gain("leaky_relu", 0.20)
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias:
                    nn.init.constant_(m.bias, 0)
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        target = datapoint["ref_image"]
        y1     = outputs["stage1"]
        y2     = outputs["enhanced"]
        loss   = 0.5 * self.loss(y1, target) + 0.5 * self.loss(y2, target)
        
        # Return
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
        x = datapoint["image"]

        # Stage 1
        x1   = self.conv_01(x)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)

        # SAM
        sam_feats, y1 = self.sam12(x1, x)

        # Stage 2
        x2     = self.conv_02(x)
        x2     = self.cat12(torch.cat([x2, sam_feats], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        y2 = self.last(x2)
        y2 = y2 + x
        
        return {
            "stage1"  : y1,
            "enhanced": y2,
        }
