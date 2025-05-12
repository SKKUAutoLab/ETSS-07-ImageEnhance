#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SIREN network with sine layers.

References:
    - https://github.com/lucidrains/siren-pytorch
"""

__all__ = [
    "SIREN",
    "SineLayer",
]

import numpy as np
import torch

from mon.nn.modules.inr import core


# ----- Sine's Layer -----
class SineLayer(torch.nn.Module):
    """Applies linear transformation with sine activation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        is_first: First layer flag for weight initialization as ``bool``.
            Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        init_weights: Initializes weights if ``True``. Default is ``True``.

    References:
        - https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w0          = w0
        self.is_first    = is_first
        self.linear      = torch.nn.Linear(in_channels, out_channels, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_channels if self.is_first else np.sqrt(6 / self.in_channels) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sine.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Sine-transformed tensor as ``torch.Tensor``.
        """
        return torch.sin(self.w0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms input and returns intermediate result.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Tuple of (sine-transformed tensor as ``torch.Tensor``,
                      intermediate tensor as ``torch.Tensor``).
        """
        intermediate = self.w0 * self.linear(x)
        return torch.sin(intermediate), intermediate


# ----- SIREN -----
class SIREN(torch.nn.Module):
    """Implements SIREN network with sine layers.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        hidden_channels: Number of channels in hidden layers as ``int``.
        hidden_layers: Number of hidden layers as ``int``.
        first_w0: Frequency for first layer as ``float``. Default is ``30.0``.
        hidden_w0: Frequency for hidden layers as ``float``. Default is ``30.0``.
        bias: Uses bias in layers if ``True``. Default is ``True``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        first_w0       : float = 30.0,
        hidden_w0      : float = 30.0,
        bias           : bool  = True,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            SineLayer(in_channels, hidden_channels, first_w0, is_first=True, bias=bias),
            *[SineLayer(hidden_channels, hidden_channels, hidden_w0, bias=bias) for _ in range(hidden_layers)],
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
