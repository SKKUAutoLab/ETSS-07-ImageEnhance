#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements custom normalization layers."""

__all__ = [
    "AdaptiveBatchNorm2d",
    "AdaptiveInstanceNorm2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "CrossMapLRN2d",
    "GroupNorm",
    "HalfInstanceNorm2d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LocalResponseNorm",
    "RMSNorm",
    "SyncBatchNorm",
]

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import *
from torch.nn.modules.instancenorm import *
from torch.nn.modules.normalization import *


# ----- Batch Normalization -----
class AdaptiveBatchNorm2d(nn.Module):
    r"""Applies adaptive Batch Normalization over a 4D input.
    
    .. math::
    
        y = w_0 \cdot x + w_1 \cdot \text{BN}(x)

    Args:
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        eps: A value added to the denominator for numerical stability. Default: ``0.999``.
        momentum: Value used for the running_mean and running_var computation.
            Can be set to ``None`` for cumulative moving average (i.e., simple average).
            Default: ``0.001``.
        kwargs: Additional keyword arguments for ``torch.nn.BatchNorm2d``.
        
    References:
        - Paper: https://arxiv.org/abs/1709.00643
        - Code: https://github.com/nrupatunga/Fast-Image-Filters
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        *args, **kwargs
    ):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.w0 * input + self.w1 * self.bn(input)


# ----- Instance Normalization -----
class AdaptiveInstanceNorm2d(nn.Module):
    r"""Applies adaptive Instance Normalization.
    
    .. math::
    
        y = w_0 \cdot x + w_1 \cdot \text{IN}(x)
    
    Args:
        num_features: Number of input channels as ``int``.
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        eps: A value added to the denominator for numerical stability. Default: ``0.999``.
        momentum: Value used for the running_mean and running_var computation.
            Default: ``0.001``.
        kwargs: Additional keyword arguments for ``torch.nn.InstanceNorm2d``.
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        *args, **kwargs
    ):
        super().__init__()
        self.w0  = nn.Parameter(torch.tensor(1.0))
        self.w1  = nn.Parameter(torch.tensor(0.0))
        self.in_ = nn.InstanceNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.w0 * input + self.w1 * self.in_(input)
    

class HalfInstanceNorm2d(nn.Module):
    r"""Applies Instance Normalization on the first half of input tensor and
    concatenates it with the second half.
    
    .. math::
        
        y = \text{IN}(x_1) \oplus x_2
    
    where :math:`\oplus` is concatenation along the channel dimension.
    
    Args:
        num_features: Number of input channels as ``int``.
        eps: Smoothing factor for stability as ``float``. Default: ``1e-5``.
        momentum: Momentum for running stats as ``float``. Default: ``0.1``.
        kwargs: Additional keyword arguments for ``torch.nn.InstanceNorm2d``.
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 1e-5,
        momentum    : float = 0.1,
        affine      : bool  = True,
        *args, **kwargs,
    ):
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"[num_features] must be even, got {num_features}.")
        self.in_ = nn.InstanceNorm2d(int(num_features // 2), eps, momentum, *args, **kwargs)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            y1, y2 = torch.chunk(input, 2, dim=0)
        else:
            y1, y2 = torch.chunk(input, 2, dim=1)
        y1 = self.in_(y1)
        y  = torch.cat([y1, y2], dim=1 if input.dim() == 4 else 0)
        return y
