#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements INR layer with various nonlinearities."""

__all__ = [
    "INRLayer",
]

from typing import Literal

import torch

from mon.nn.modules.inr.core import ReLULayer, SigmoidLayer, TanhLayer
from mon.nn.modules.inr.finer import FINERLayer
from mon.nn.modules.inr.gauss import GaussLayer
from mon.nn.modules.inr.siren import SineLayer
from mon.nn.modules.inr.wire import ComplexGaborLayer

INR_AF = Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"]


# ----- General INR Layer -----
class INRLayer(torch.nn.Module):
    """Combines linear transformation, nonlinearity, and dropout.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        nonlinear: Nonlinearity type as ``Literal["sigmoid", "tanh", "relu", "sine",
            "gauss", "wire", "finer"]``. Default is ``"sine"``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        scale: Gaussian scale factor as ``float``. Default is ``10.0``.
        first_bias_scale: Bias scale for first "finer" layer as ``float`` or ``None``.
            Default is ``None``.
        is_first: First layer flag as ``bool``. Default is ``False``.
        is_last: Last layer flag, forces "sigmoid", as ``bool``. Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        nonlinear       : Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"] = "sine",
        w0              : float = 30.0,
        scale           : float = 10.0,
        first_bias_scale: float = None,
        is_first        : bool  = False,
        is_last         : bool  = False,
        bias            : bool  = True,
        dropout         : float = 0.0,
    ):
        super().__init__()
        if is_last:
            nonlinear = "sigmoid"
        
        layer_args = {
            "in_channels" : in_channels,
            "out_channels": out_channels,
            "bias"        : bias
        }
        
        if nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(**layer_args)
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(**layer_args)
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(**layer_args)
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(**layer_args, w0=w0, is_first=is_first, init_weights=not is_last)
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(**layer_args, scale=scale)
        elif nonlinear == "wire":
            self.nonlinear = ComplexGaborLayer(**layer_args, w0=w0, is_first=is_first)
        elif nonlinear == "finer":
            self.nonlinear = FINERLayer(**layer_args, w0=w0, first_bias_scale=first_bias_scale, is_first=is_first)
        else:
            raise ValueError(f"[nonlinear] must be supported type, got {nonlinear}.")
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies nonlinearity and dropout to input.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        return self.dropout(self.nonlinear(x))
