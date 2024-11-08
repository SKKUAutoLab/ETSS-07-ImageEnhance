#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implicit Neural Representations.

This module implements Implicit Neural Representations (INR), its variants and
networks.

References:
    https://github.com/lucidrains/siren-pytorch
    https://github.com/vishwa91/wire
"""

from __future__ import annotations

__all__ = [
    "ContextImplicitCoordinatesEncoder",
    "ContextImplicitDecoder",
    "ContextImplicitFeatureEncoder",
    "FINER",
    "GAUSS",
    "INRLayer",
    "INRModulatorWrapper",
    "PEMLP",
    "SIREN",
    "WIRE",
]

from typing import Literal, Sequence

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from mon import core
from mon.nn.modules import activation as act


# region Utils

def get_coords(size: int | Sequence[int]) -> torch.Tensor:
    """Creates a coordinates grid.
    
    Args:
        size: The size of the coordinates grid.
    """
    size   = core.get_image_size(size)
    h, w   = size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()
    return coords

# endregion


# region INR Layer

class ComplexGaborLayer(nn.Module):
    """Complex Gabor Layer from WIRE (https://github.com/vishwa91/wire)
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 10.0,
        sigma_0     : float = 40.0,
        trainable   : bool  = False
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.scale_0     = sigma_0
        self.is_first    = is_first
        self.in_channels = in_channels
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        self.linear  = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin   = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


class FINERLayer(nn.Module):
    """FINER Layer.
    
    For the value of ``first_bias_scale``, see Fig. 5 in the paper.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        first_bias_scale: The scale of the first bias. Defaults: ``20.0``.
        scale_req_grad: Whether the scale requires gradient. Defaults: ``False``.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        bias            : bool  = True,
        is_first        : bool  = False,
        omega_0         : float = 30,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        
        self.init_weights()
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels,
                                             1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
    
    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.linear(x)
        scale = self.generate_scale(x)
        out   = torch.sin(self.omega_0 * scale * x)
        return out


class GaussLayer(nn.Module):
    """Drop in replacement for SineLayer but with Gaussian non-linearity
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/gauss.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool  = True,
        scale       : float = 10.0,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scale       = scale
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        N_freqs    : int,
        logscale   : bool = True,
    ):
        super().__init__()
        self.N_freqs      = N_freqs
        self.in_channels  = in_channels
        self.funcs        = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)
    

class ReLULayer(nn.Module):
    """Drop in replacement for SineLayer but with ReLU non-linearity
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/relu.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        scale       : float = 10.0,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.omega_0     = omega_0
        self.scale       = scale
        self.is_first    = is_first
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self.linear(x))
 

class SigmoidLayer(nn.Module):
    """Drop in replacement for SineLayer but with Sigmoid non-linearity.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
        init_weights: Whether to initialize the weights. Defaults: ``True``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        self.act         = act.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class SineLayer(nn.Module):
    """Sine Layer.
    
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    discussion of omega_0.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
        init_weights: Whether to initialize the weights. Defaults: ``True``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        scale       : float = 10.0,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.omega_0     = omega_0
        self.scale       = scale
        self.is_first    = is_first
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels,
                                             1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class TanhLayer(nn.Module):
    """Drop in replacement for SineLayer but with Tanh non-linearity.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
        init_weights: Whether to initialize the weights. Defaults: ``True``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        self.act         = act.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
    

class INRLayer(nn.Module):
    """INR Layer with different nonlinear layers. The layer consists of:
    (linear + non-linear) + dropout.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        is_last: Whether this is the last layer. Defaults: ``False``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
        first_bias_scale: The scale of the first bias. Defaults: ``20.0``.
        nonlinear: The non-linearity to use. The layer defined here already
            include a ``nn.Linear()`` layer. One of: ``"gauss"``, ``"relu"``,
            ``"sigmoid"``, ``"sine"``, ``"finer"``. Defaults: ``"sine"``.
        dropout: The dropout rate. Defaults: ``0.0``.
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        bias            : bool  = True,
        is_first        : bool  = False,
        is_last         : bool  = False,
        omega_0         : float = 30.0,
        scale           : float = 10.0,
        first_bias_scale: float = None,
        nonlinear       : Literal["gauss", "finer", "relu", "sigmoid", "sine", "tanh"] = "sine",
        dropout         : float = 0.0
    ):
        super().__init__()
        if is_last:
            nonlinear = "sigmoid"
        
        if nonlinear == "finer":
            self.nonlinear = FINERLayer(
                in_channels      = in_channels,
                out_channels     = out_channels,
                bias             = bias,
                is_first         = is_first,
                omega_0          = omega_0,
                scale_req_grad   = False,
                first_bias_scale = first_bias_scale,
            )
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
                scale        = scale,
            )
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
                is_first     = is_first,
                omega_0      = omega_0,
                scale        = scale,
            )
        elif nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
            )
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
                is_first     = is_first,
                omega_0      = omega_0,
                scale        = scale,
                init_weights = not is_last,
            )
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
            )
        else:
            raise ValueError(f"Non-linearity '{nonlinear}' is not supported.")
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.nonlinear(x)
        y = self.dropout(y)
        return y


class INRModulator(nn.Module):
    
    def __init__(self, in_channels: int, hidden_channels: int, hidden_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for ind in range(hidden_layers):
            is_first    = ind == 0
            in_channels = in_channels if is_first else (hidden_channels + in_channels)
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                )
            )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x       = z
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))
        return tuple(hiddens)


class INRModulatorWrapper(nn.Module):
    
    def __init__(
        self,
        net            : nn.Module,
        image_width    : int = 256,
        image_height   : int = 256,
        latent_channels: int = None
    ):
        super().__init__()
        self.net          = net
        self.image_width  = image_width
        self.image_height = image_height

        self.modulator = None
        if latent_channels is not None:
            self.modulator = INRModulator(
                in_channels     = latent_channels,
                hidden_channels = net.hidden_channels,
                hidden_layers   = net.hidden_layers
            )

        tensors = [torch.linspace(-1, 1, steps=image_height), torch.linspace(-1, 1, steps=image_width)]
        mgrid   = torch.stack(torch.meshgrid(*tensors, indexing = "ij"), dim=-1)
        mgrid   = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = self.modulator is not None
        assert not (modulate ^ latent is not None), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'
        mods   = self.modulator(latent) if modulate else None
        coords = self.grid.clone().detach().requires_grad_()
        out    = self.net(coords, mods)
        out    = rearrange(out, "(h w) c -> () c h w", h=self.image_height, w=self.image_width)
        if img is not None:
            return F.mse_loss(img, out)
        return out
    
# endregion


# region FINER Network

class FINER(nn.Module):
    """FINER network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        hidden_channels : int,
        hidden_layers   : int,
        bias            : bool  = True,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [FINERLayer(in_channels, hidden_channels, bias, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad)]
        for i in range(hidden_layers):
            net.append(FINERLayer(hidden_channels, hidden_channels, is_first=False, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))
        
        final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = core.get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)

# endregion


# region GAUSS Network

class GAUSS(nn.Module):
    """Gauss network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        bias           : bool  = True,
        scale          : float = 30.0
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [GaussLayer(in_channels, hidden_channels, bias, scale)]
        for i in range(hidden_layers):
            net.append(GaussLayer(hidden_channels, hidden_channels, bias, scale))
        final_linear = nn.Linear(hidden_channels, out_channels)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = core.get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)

# endregion


# region PEMLP Network

class PEMLP(nn.Module):
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        N_freqs        : int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        self.encoding        = PositionalEncoding(in_channels=in_channels, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.encoding.out_channels, hidden_channels))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_channels, hidden_channels))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_channels, out_channels)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = core.get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)


# endregion


# region SIREN Network

class SIREN(nn.Module):
    """SIREN network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        bias           : bool  = True,
        first_omega_0  : float = 30.0,
        hidden_omega_0 : float = 30.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [SineLayer(in_channels, hidden_channels, bias, is_first=True, omega_0=first_omega_0)]
        for i in range(hidden_layers):
            net.append(SineLayer(hidden_channels, hidden_channels, is_first=False, omega_0=hidden_omega_0))
        
        final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = core.get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)

# endregion


# region WIRE Network

class WIRE(nn.Module):
    """WIRE network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        bias           : bool  = True,
        first_omega_0  : float = 20,
        hidden_omega_0 : float = 20,
        scale          : float = 10.0
    ):
        super().__init__()
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_channels = int(hidden_channels / np.sqrt(2))
        dtype           = torch.cfloat
        
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        self.complex         = True
        
        net = [ComplexGaborLayer(in_channels, hidden_channels, bias, is_first=True, omega_0=first_omega_0, sigma_0=scale)]
        for i in range(hidden_layers):
            net.append(ComplexGaborLayer(hidden_channels, hidden_channels, is_first=False, omega_0=hidden_omega_0, sigma_0=scale))
        
        final_linear = nn.Linear(hidden_channels, out_channels, dtype=dtype)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = core.get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)

# endregion


# region Context-INR

class ContextImplicitFeatureEncoder(nn.Module):
    """Implicit Neural Representation (INR) of a specific value of a
    context-window in an image.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size      : int   = 1,
        out_channels     : int   = 256,
        down_size        : int   = 256,
        hidden_layers    : int   = 2,
        omega_0          : float = 30.0,
        first_bias_scale : float = None,
        nonlinear        : Literal["finer", "gauss", "relu", "sigmoid", "sine"] = "sine",
        weight_decay     : float = 0.0001,
        use_ff           : bool  = False,
        ff_gaussian_scale: float = 10,
    ):
        super().__init__()
        self.window_size     = window_size
        self.down_size       = down_size
        self.hidden_channels = out_channels
        self.hidden_layers   = hidden_layers
        in_channels          = window_size ** 2
        net_in_channels      = in_channels
        
        if use_ff:
            self.register_buffer("B", torch.randn((out_channels, in_channels)) * ff_gaussian_scale)
            net_in_channels = out_channels * 2
        else:
            self.B = None
        
        net = [INRLayer(net_in_channels, out_channels, is_first=True, omega_0=omega_0, first_bias_scale=first_bias_scale, nonlinear=nonlinear)]
        for _ in range(1, hidden_layers):
            net.append(INRLayer(out_channels, out_channels, is_first=False, omega_0=omega_0, nonlinear=nonlinear))
        net.append(INRLayer(out_channels, out_channels, is_first=False, omega_0=omega_0, nonlinear=nonlinear))
        self.net = nn.Sequential(*net)
        
        weight_decay = weight_decay or 0.0001
        self.params  = [{"params": self.net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_lr  = self.interpolate_image(image)
        patch     = self.get_patch(image_lr)
        embedding = self.ff_embedding(patch, self.B)
        patch     = self.net(embedding)
        return image_lr, patch
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patch(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(image.device)
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
        
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)
    
    def ff_embedding(self, image: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        if B is None:
            return image
        else:
            x_proj    = (2. * np.pi * image) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding
        

class ContextImplicitCoordinatesEncoder(nn.Module):
    """Implicit Neural Representation (INR) of coordinates (x, y) of a
    context-window an image.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        out_channels     : int   = 256,
        down_size        : int   = 256,
        hidden_layers    : int   = 2,
        omega_0          : float = 30.0,
        first_bias_scale : float = None,
        nonlinear        : Literal["finer", "gauss", "relu", "sigmoid", "sine"] = "sine",
        weight_decay     : float = 0.1,
        use_ff           : bool  = False,
        ff_gaussian_scale: float = 10,
    ):
        super().__init__()
        self.down_size       = down_size
        self.hidden_channels = out_channels
        self.hidden_layers   = hidden_layers
        in_channels          = 2
        hidden_channels      = in_channels
        
        if use_ff:
            self.register_buffer("B", torch.randn((out_channels, in_channels)) * ff_gaussian_scale)
            hidden_channels = out_channels * 2
        else:
            self.B = None
            
        net = [INRLayer(hidden_channels, out_channels, is_first=True, omega_0=omega_0, first_bias_scale=first_bias_scale, nonlinear=nonlinear)]
        for _ in range(1, hidden_layers):
            net.append(INRLayer(out_channels, out_channels, is_first=False, omega_0=omega_0, nonlinear=nonlinear))
        net.append(INRLayer(out_channels, out_channels, is_first=False, omega_0=omega_0, nonlinear=nonlinear))
        self.net = nn.Sequential(*net)
        
        weight_decay = weight_decay or 0.1
        self.params  = [{"params": self.net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        coords    = get_coords((self.down_size, self.down_size)).to(image.device)
        embedding = self.ff_embedding(coords, self.B)
        coords    = self.net(embedding)
        return coords
    
    def ff_embedding(self, image: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        if B is None:
            return image
        else:
            x_proj    = (2. * np.pi * image) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding
  

class ContextImplicitDecoder(nn.Module):
    """MLP for combining values and coordinates INRs.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        in_channels  : int   = 256,
        out_channels : int   = 3,
        hidden_layers: int   = 1,
        omega_0      : float = 30.0,
        nonlinear    : Literal["gauss", "relu", "sigmoid", "sine", "finer"] = "sine",
        weight_decay : float = 0.001,
    ):
        super().__init__()
        self.hidden_channels = out_channels
        self.hidden_layers   = hidden_layers
        
        net = []
        for _ in range(0, hidden_layers):
            net.append(INRLayer(in_channels, in_channels, is_first=False, omega_0=omega_0, nonlinear=nonlinear))
        net.append(INRLayer(in_channels, out_channels, is_last=True, omega_0=omega_0))
        self.net = nn.Sequential(*net)
        
        weight_decay = weight_decay or 0.001
        self.params  = [{"params": self.net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# endregion
