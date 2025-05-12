#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FINER network with FINER layers."""

__all__ = [
    "FINER",
    "FINERLayer",
]

import numpy as np
import torch

from mon.nn.modules.inr import core


# ----- FINER's Activation Layer -----
class FINERLayer(torch.nn.Module):
    """Applies scaled sine activation to linear transformation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float``. Default is ``20.0``.
        is_first: First layer flag for initialization as ``bool``. Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default is ``False``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.w0               = w0
        self.is_first         = is_first
        self.in_channels      = in_channels
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        self.linear           = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.init_weights()
        if self.first_bias_scale and self.is_first:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_channels if self.is_first else np.sqrt(6 / self.in_channels) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor for activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Scaling tensor as ``torch.Tensor``.
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with scaled sine activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.w0 * scale * linear)


# ----- FINER -----
class FINER(torch.nn.Module):
    """Implements FINER network with FINER layers.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        hidden_channels: Number of channels in hidden layers as ``int``.
        hidden_layers: Number of hidden layers as ``int``.
        first_w0: Frequency for first layer as ``float``. Default is ``30.0``.
        hidden_w0: Frequency for hidden layers as ``float``. Default is ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float`` or ``None``.
            Default is ``None``.
        bias: Uses bias in layers if ``True``. Default is ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default is ``False``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        hidden_channels : int,
        hidden_layers   : int,
        first_w0        : float = 30.0,
        hidden_w0       : float = 30.0,
        first_bias_scale: float = None,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            FINERLayer(in_channels, hidden_channels, first_w0, first_bias_scale, is_first=True, bias=bias, scale_req_grad=scale_req_grad),
            *[FINERLayer(hidden_channels, hidden_channels, hidden_w0, bias=bias, scale_req_grad=scale_req_grad) for _ in range(hidden_layers)],
            torch.nn.Linear(hidden_channels, out_channels)
        )
        with torch.no_grad():
            self.net[-1].weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_w0, np.sqrt(6 / hidden_channels) / hidden_w0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor as ``torch.Tensor`` for size reference.

        Returns:
            Output tensor as ``torch.Tensor`` from network.
        """
        from mon import vision
        s, _   = vision.image_size(x)
        coords = core.create_coords(s).to(x.device)
        return self.net(coords)
