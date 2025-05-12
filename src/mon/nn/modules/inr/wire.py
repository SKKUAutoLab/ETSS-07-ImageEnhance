#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements WIRE network with complex Gabor layers.

References:
    - https://github.com/vishwa91/wire
"""

__all__ = [
    "WIRE",
    "ComplexGaborLayer",
]

import numpy as np
import torch

from mon.nn.modules.inr import core


# ----- Complex Gabor Layer -----
class ComplexGaborLayer(torch.nn.Module):
    """Applies complex Gabor transformation to input.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        w0: Base frequency factor as ``float``. Default is ``10.0``.
        s0: Base scale factor as ``float``. Default is ``40.0``.
        is_first: First layer flag for dtype as ``bool``. Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        trainable: Parameters trainable if ``True``. Default is ``False``.

    References:
        - https://github.com/vishwa91/wire
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 10.0,
        s0          : float = 40.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        trainable   : bool  = False
    ):
        super().__init__()
        self.is_first    = is_first
        self.in_channels = in_channels
        dtype            = torch.float if is_first else torch.cfloat
        self.linear      = torch.nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype)
        self.w0          = torch.nn.Parameter(torch.tensor([w0]), requires_grad=trainable)
        self.scale_0     = torch.nn.Parameter(torch.tensor([s0]), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with complex Gabor activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Complex-valued transformed tensor as ``torch.Tensor``.
        """
        linear = self.linear(x)
        omega  = self.w0 * linear
        scale  = self.scale_0 * linear
        return torch.exp(1j * omega - scale.abs().square())


# ----- WIRE -----
class WIRE(torch.nn.Module):
    """WIRE network.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        hidden_channels: Number of channels in hidden layers as ``int`` (adjusted internally).
        hidden_layers: Number of hidden layers as ``int``.
        first_w0: Frequency for first layer as ``float``. Default is ``20.0``.
        hidden_w0: Frequency for hidden layers as ``float``. Default is ``20.0``.
        scale: Gaussian scale factor as ``float``. Default is ``10.0``.
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
        first_w0       : float = 20,
        hidden_w0      : float = 20,
        scale          : float = 10.0,
        bias           : bool  = True,
    ):
        super().__init__()
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_channels = int(hidden_channels / np.sqrt(2))
        dtype = torch.cfloat

        self.net = torch.nn.Sequential(
            ComplexGaborLayer(in_channels, hidden_channels, first_w0, s0=scale, is_first=True, bias=bias),
            *[ComplexGaborLayer(hidden_channels, hidden_channels, hidden_w0, s0=scale, bias=bias) for _ in range(hidden_layers)],
            torch.nn.Linear(hidden_channels, out_channels, dtype=dtype)
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
