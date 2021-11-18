#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pooling Layers.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish

from torchkit.core.utils import Size2T
from torchkit.core.utils import Size4T
from torchkit.core.utils import to_2tuple
from torchkit.core.utils import to_4tuple
from .builder import POOL_LAYERS
from .conv_norm_act import ConvBnMish
from .padding import get_padding
from .padding import get_padding_value
from .padding import pad_same

logger = logging.getLogger()


# MARK: - Builder

POOL_LAYERS.register(name="adaptive_avg_pool",      module=nn.AdaptiveAvgPool2d)
POOL_LAYERS.register(name="adaptive_avg_pool_1d",   module=nn.AdaptiveAvgPool1d)
POOL_LAYERS.register(name="adaptive_avg_pool_2d",   module=nn.AdaptiveAvgPool2d)
POOL_LAYERS.register(name="adaptive_avg_pool_3d",   module=nn.AdaptiveAvgPool3d)
POOL_LAYERS.register(name="adaptive_max_pool",      module=nn.AdaptiveMaxPool2d)
POOL_LAYERS.register(name="adaptive_max_pool_1d",   module=nn.AdaptiveMaxPool1d)
POOL_LAYERS.register(name="adaptive_max_pool_2d",   module=nn.AdaptiveMaxPool2d)
POOL_LAYERS.register(name="adaptive_max_pool_3d",   module=nn.AdaptiveMaxPool3d)
POOL_LAYERS.register(name="avg_pool",		        module=nn.AvgPool2d)
POOL_LAYERS.register(name="avg_pool_1d",		    module=nn.AvgPool1d)
POOL_LAYERS.register(name="avg_pool_2d",		    module=nn.AvgPool2d)
POOL_LAYERS.register(name="avg_pool_3d", 		    module=nn.AvgPool3d)
POOL_LAYERS.register(name="fractional_max_pool_2d", module=nn.FractionalMaxPool2d)
POOL_LAYERS.register(name="fractional_max_pool_3d", module=nn.FractionalMaxPool3d)
POOL_LAYERS.register(name="lp_pool_1d", 			module=nn.LPPool1d)
POOL_LAYERS.register(name="lp_pool_2d", 			module=nn.LPPool2d)
POOL_LAYERS.register(name="max_pool", 				module=nn.MaxPool2d)
POOL_LAYERS.register(name="max_pool_1d", 			module=nn.MaxPool1d)
POOL_LAYERS.register(name="max_pool_2d", 			module=nn.MaxPool2d)
POOL_LAYERS.register(name="max_pool_3d", 			module=nn.MaxPool3d)
POOL_LAYERS.register(name="max_unpool", 			module=nn.MaxUnpool2d)
POOL_LAYERS.register(name="max_unpool_1d", 			module=nn.MaxUnpool1d)
POOL_LAYERS.register(name="max_unpool_2d", 			module=nn.MaxUnpool2d)
POOL_LAYERS.register(name="max_unpool_3d", 			module=nn.MaxUnpool3d)


# MARK: - AvgPool2dSame

def avg_pool2d_same(
    x                : torch.Tensor,
    kernel_size      : list[int],
    stride           : list[int],
    padding          : list[int] = (0, 0),
    ceil_mode        : bool      = False,
    count_include_pad: bool      = True
) -> torch.Tensor:
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(
        x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad
    )


@POOL_LAYERS.register(name="avg_pool_2d_same")
class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'same' wrapper for 2D average pooling."""

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size      : Size2T,
        stride           : Optional[Size2T]         = None,
        padding          : Union[Size2T, str, None] = 0,
        ceil_mode        : bool                     = False,
        count_include_pad: bool                     = True
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        super().__init__(
            kernel_size, stride, (0, 0), ceil_mode, count_include_pad
        )

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


AvgPoolSame = AvgPool2dSame
POOL_LAYERS.register(name="avg_pool_same", module=AvgPoolSame)


# MARK: - BlurPool2d

class BlurPool2d(nn.Module):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details. Corresponds to the
    Downsample class, which does blurring and subsampling
    
    Args:
        channels (int):
            Number of input channels
        filter_size (int):
            Binomial filter size for blurring. currently supports 3 and 5.
            Default: `3`.
        stride (int):
            Downsampling filter stride.
            
    Returns:
        x (torch.Tensor):
            The transformed tensor.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self, channels: int, filter_size: int = 3, stride=2
    ):
        super(BlurPool2d, self).__init__()
        assert filter_size > 1
        self.channels    = channels
        self.filter_size = filter_size
        self.stride      = stride
        self.padding     = [get_padding(filter_size, stride, dilation=1)] * 4
        
        poly1d = np.poly1d((0.5, 0.5))
        coeffs = torch.tensor(
            (poly1d ** (self.filt_size - 1)).coeffs.astype(np.float32)
        )
        blur_filter = (coeffs[:, None] * coeffs[None, :]
                       )[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer("filter", blur_filter, persistent=False)
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, "reflect")
        return F.conv2d(x, self.filt, stride=self.stride, groups=x.shape[1])
    

# MARK: - MaxPool2dSame

def max_pool2d_same(
    x          : torch.Tensor,
    kernel_size: list[int],
    stride     : list[int],
    padding    : list[int] = (0, 0),
    dilation   : list[int] = (1, 1),
    ceil_mode  : bool      = False
) -> torch.Tensor:
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


@POOL_LAYERS.register(name="max_pool_2d_same")
class MaxPool2dSame(nn.MaxPool2d):
    """Tensorflow like `same` wrapper for 2D max pooling."""

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size: Size2T,
        stride     : Optional[Size2T]         = None,
        padding    : Union[Size2T, str, None] = 0,
        dilation   : Size2T                   = (1, 1),
        ceil_mode  : bool                     = False
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        super().__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0),
                            self.dilation, self.ceil_mode)


MaxPoolSame = MaxPool2dSame
POOL_LAYERS.register(name="max_pool_same", module=MaxPoolSame)


# MARK: - MedianPool2d

@POOL_LAYERS.register(name="median_pool_2d")
class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Attributes:
         kernel_size (Size2T):
            Size of pooling kernel.
         stride (Size2T):
            Pool stride, int or 2-tuple
         padding (Size4T, str, optional):
            Pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad.
         same (bool):
            Override padding and enforce same padding. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size: Size2T    				  = (3, 3),
        stride     : Size2T    			      = (1, 1),
        padding    : Union[Size4T, str, None] = 0,
        same	   : bool					  = False
    ):
        super().__init__()
        self.kernel_size = to_2tuple(kernel_size)
        self.stride 	 = to_2tuple(stride)
        self.padding 	 = to_4tuple(padding)  # convert to l, r, t, b
        self.same	 	 = same

    # MARK: Configure

    def _padding(self, x: torch.Tensor):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.kernel_size[0] - self.stride[0], 0)
            else:
                ph = max(self.kernel_size[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.kernel_size[1] - self.stride[1], 0)
            else:
                pw = max(self.kernel_size[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0])
        x = x.unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


MedianPool = MedianPool2d
POOL_LAYERS.register(name="median_pool", module=MedianPool)


# MARK: - SpatialPyramidPooling

@POOL_LAYERS.register(name="spatial_pyramid_pool")
class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (tuple):
            Sizes of several convolving kernels. Default: `(5, 9, 13)`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : tuple = (5, 9, 13),
    ):
        super().__init__()
        hidden_channels = in_channels // 2  # Hidden channels
        in_channels2    = hidden_channels * (len(kernel_size) + 1)

        self.conv1 = ConvBnMish(in_channels, hidden_channels, kernel_size=1,
                            stride=1)
        self.conv2 = ConvBnMish(in_channels2, out_channels, kernel_size=1,
                            stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2)
             for x in kernel_size]
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.conv1(x)
        y_hat = self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return y_hat


SPP = SpatialPyramidPooling
POOL_LAYERS.register(name="spp", module=SPP)


# MARK: - SpatialPyramidPooling CrossStagePartial

@POOL_LAYERS.register(name="spatial_pyramid_pooling_csp")
class SpatialPyramidPoolingCSP(nn.Module):
    """Cross Stage Partial Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        number (int):
            Number of bottleneck layers to use.
        shortcut (bool):
            Use shortcut connection?. Default: `True`.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
        kernel_size (tuple):
            Sizes of several convolving kernels. Default: `(5, 9, 13)`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = False,
        groups      : int   = 1,
        expansion   : float = 0.5,
        kernel_size : tuple = (5, 9, 13),
    ):
        super().__init__()
        hidden_channels = int(2 * out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish(in_channels, hidden_channels, kernel_size=1,
                                 stride=1)
        self.conv2 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1),
            bias=False
        )
        self.conv3 = ConvBnMish(hidden_channels, hidden_channels,
                                kernel_size=3, stride=1)
        self.conv4 = ConvBnMish(hidden_channels, hidden_channels,
                                kernel_size=1, stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=(1, 1), padding=x // 2)
             for x in kernel_size]
        )
        self.conv5 = ConvBnMish(4 * hidden_channels, hidden_channels,
                                kernel_size=1, stride=1)
        self.conv6 = ConvBnMish(hidden_channels, hidden_channels,
                                kernel_size=3, stride=1)
        self.bn    = nn.BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.conv7 = ConvBnMish(2 * hidden_channels, hidden_channels,
                                kernel_size=1, stride=1)
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(
            self.conv5(torch.cat([x1] + [m(x1) for m in self.m], 1))
        )
        y2 = self.conv2(x)
        y_hat = self.conv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        return y_hat


SPPCSP = SpatialPyramidPoolingCSP
POOL_LAYERS.register(name="spp_csp", module=SPPCSP)


# MARK: - Builder

def create_pool2d(
    pool_type  : str,
    kernel_size: Size2T,
    stride	   : Optional[Size2T] = None,
    **kwargs
):
    stride              = stride or kernel_size
    padding             = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
		padding, kernel_size, stride=stride, **kwargs
	)

    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride, padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride, padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
