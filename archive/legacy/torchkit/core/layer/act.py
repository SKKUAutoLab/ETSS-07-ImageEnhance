#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common Activation Layers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.core.utils import FuncCls
from .builder import ACT_LAYERS

logger = logging.getLogger()


# MARK: - Register

ACT_LAYERS.register(name="celu",         module=nn.CELU)
ACT_LAYERS.register(name="elu",          module=nn.ELU)
# ACT_LAYERS.register(name="gelu",         module=nn.GELU)
ACT_LAYERS.register(name="hard_shrink",  module=nn.Hardshrink)
# ACT_LAYERS.register(name="hard_sigmoid", module=nn.Hardsigmoid)
# ACT_LAYERS.register(name="hard_swish", 	 module=nn.Hardswish)
ACT_LAYERS.register(name="hard_tanh",    module=nn.Hardtanh)
ACT_LAYERS.register(name="identity",     module=nn.Identity)
ACT_LAYERS.register(name="leaky_relu",   module=nn.LeakyReLU)
ACT_LAYERS.register(name="log_sigmoid",  module=nn.LogSigmoid)
ACT_LAYERS.register(name="log_softmax",  module=nn.LogSoftmax)
# ACT_LAYERS.register(name="prelu",        module=nn.PReLU)
ACT_LAYERS.register(name="relu", 		 module=nn.ReLU)
ACT_LAYERS.register(name="relu6", 		 module=nn.ReLU6)
ACT_LAYERS.register(name="rrelu", 		 module=nn.RReLU)
ACT_LAYERS.register(name="selu", 		 module=nn.SELU)
# ACT_LAYERS.register(name="sigmoid",		 module=nn.Sigmoid)
ACT_LAYERS.register(name="silu", 		 module=nn.SiLU)
ACT_LAYERS.register(name="softmax",      module=nn.Softmax)
ACT_LAYERS.register(name="softmin",      module=nn.Softmin)
ACT_LAYERS.register(name="softplus", 	 module=nn.Softplus)
ACT_LAYERS.register(name="softshrink",   module=nn.Softshrink)
ACT_LAYERS.register(name="softsign",     module=nn.Softsign)
# ACT_LAYERS.register(name="tanh",		 module=nn.Tanh)
ACT_LAYERS.register(name="tanhshrink",   module=nn.Tanhshrink)


# MARK: - ArgMax

@ACT_LAYERS.register(name="arg_max")
class ArgMax(nn.Module):
    """Find the indices of the maximum value of all elements in the input
    tensor.
    
    Attributes:
        dim (int, optional):
            The dimension to find the indices of the maximum value.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=self.dim)


# MARK: - Clamp/Clip

@ACT_LAYERS.register(name="clamp")
class Clamp(nn.Module):
    """Clamp activation layer. This activation function is to clamp the feature
    map value within :math:`[min, max]`. More details can be found in
    `torch.clamp()`.
    
    Attributes:
        min (float):
            Lower-bound of the range to be clamped to.
        max (float):
            Upper-bound of the range to be clamped to.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, min: float = -1.0, max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min, max=self.max)


Clip = Clamp
ACT_LAYERS.register(name="clip", module=Clip)


# MARK: - GELU

def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return F.gelu(x)


@ACT_LAYERS.register(name="gelu")
class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg).
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x, self.inplace)


# MARK: - HardMish

def hard_mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Hard Mish Experimental, based on notes by Mish author Diganta Misra at
    https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


@ACT_LAYERS.register(name="hard_mish")
class HardMish(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_mish(x, self.inplace)
    

# MARK: - HardSigmoid

def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


@ACT_LAYERS.register(name="hard_sigmoid")
class HardSigmoid(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_sigmoid(x, self.inplace)


# MARK: - HardSwish

def hard_swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


@ACT_LAYERS.register(name="hard_swish")
class HardSwish(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x, self.inplace)
    

# MARK: - Mish

def mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


@ACT_LAYERS.register(name="mish")
class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


# MARK: - PReLU

@ACT_LAYERS.register(name="prelu")
class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_parameters: int   = 1,
        init          : float = 0.25,
        inplace       : bool  = False
    ):
        super().__init__(num_parameters=num_parameters, init=init)
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.weight)


# MARK: - Sigmoid

def sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return x.sigmoid_() if inplace else x.sigmoid()


@ACT_LAYERS.register(name="sigmoid")
class Sigmoid(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid_() if self.inplace else x.sigmoid()


# MARK: - Swish

def swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Swish described in: https://arxiv.org/abs/1710.05941"""
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


@ACT_LAYERS.register(name="swish")
class Swish(nn.Module):
    """Swish Module. This module applies the swish function."""
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, self.inplace)


# MARK: - Tanh

def tanh(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return x.tanh_() if inplace else x.tanh()


@ACT_LAYERS.register(name="tanh")
class Tanh(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh_() if self.inplace else x.tanh()


# MARK: - Builder

def create_act_layer(
    apply_act: bool              = True,
    act_layer: Optional[FuncCls] = nn.ReLU,
    inplace  : bool              = True,
    **_
) -> nn.Module:
    """Create activation layer."""
    if isinstance(act_layer, str):
        act_layer = ACT_LAYERS.build(name=act_layer)
    if (act_layer is not None) and apply_act:
        act_args  = dict(inplace=True) if inplace else {}
        act_layer = act_layer(**act_args)
    else:
        act_layer = nn.Identity()
    return act_layer
