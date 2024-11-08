#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EDSR Encoder.

This module implements the encoder part of EDSR. The EDSR is introduced in the
paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution,"
Lim et al. (2017).

References:
    Modified from: https://github.com/thstkdgus35/EDSR-PyTorch
    
Pretrained Weights:
    'r16f64x2' : 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3' : 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4' : 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
"""

from __future__ import annotations

__all__ = [
    "EDSREncoder",
]

import math

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


# region EDSREncoder

def default_conv(in_channels: int, out_channels: int, kernel_size: _size_2_t, bias: bool = True) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    
    def __init__(
        self,
        rgb_range,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std : tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign    : int = -1
    ):
        super().__init__(3, 3, kernel_size=1)
        std              = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data   = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    
    def __init__(
        self,
        conv,
        n_feats    : int,
        kernel_size: _size_2_t,
        bias       : bool = True,
        bn         : bool = False,
        act               = nn.ReLU(True),
        res_scale  : int  = 1
    ):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body      = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res  = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    
    def __init__(
        self,
        conv,
        scale  : int,
        n_feats: int,
        bn     : bool       = False,
        act    : str | bool = False,
        bias   : bool       = True
    ):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSREncoder(nn.Module):
    """EDSR Encoder.
    
    Args:
        conv: Convolution layer. Default is ``default_conv``.
        in_channels: Number of input channels. Default is ``3``.
        n_resblocks: Number of residual blocks. One of: ``16`` or ``32``.
            Default: ``16``.
        n_feats: Number of features. One of: ``64`` or ``256``.
            Default: ``64``.
        res_scale: Residual scaling. Default: ``1``.
        scale: Upsampling scale. One of: ``2``, ``3``, or ``4``. Default: ``2``.
        no_upsampling: No upsampling. Default: ``True`` else the model will run
            as a normal super-resolution model.
        rgb_range: RGB range (``1`` or ``255``). Default: ``1``.
    """
    
    zoo: dict = {
        "r16f64x2": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
        },
        "r16f64x3": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
        },
        "r16f64x4": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
        },
        "r32f256x2": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
        },
        "r32f256x3": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
        },
        "r32f256x4": {
            "url": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
        },
    }
    
    def __init__(
        self,
        conv         : nn.Module = default_conv,
        in_channels  : int       = 3,
        n_resblocks  : int       = 16,
        n_feats      : int       = 64,
        res_scale    : float     = 1,
        scale        : int       = 2,
        no_upsampling: bool      = True,
        rgb_range    : int       = 1
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.no_upsampling = no_upsampling
        kernel_size        = 3
        act                = nn.ReLU(True)
        
        # Define pretrained weights URL
        url_name    = "r{}f{}x{}".format(n_resblocks, n_feats, scale)
        if url_name in self.zoo:
            self.url = self.zoo[url_name]["url"]
        else:
            self.url = None
            
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # Define head module
        m_head = [conv(self.in_channels, n_feats, kernel_size)]
        
        # Define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        
        # Define tail module
        if no_upsampling:
            self.out_channels = n_feats
        else:
            self.out_channels = self.in_channels
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, self.in_channels, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.sub_mean(x)
        x    = self.head(x)
        res  = self.body(x)
        res += x
        if self.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        # x = self.add_mean(x)s
        return x
    
    def load_state_dict(self, state_dict, strict: bool = True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}."
                            .format(name, own_state[name].size(), param.size())
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError("unexpected key '{}' in state_dict".format(name))
                
# endregion
