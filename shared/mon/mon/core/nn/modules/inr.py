#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Implicit Neural Representations (INR) layers and networks."""

__all__ = [
    "FINER",
    "FINERLayer",
    "GAUSS",
    "GaussLayer",
    "INRLayer",
    "LinearLayer",
    "ReLULayer",
    "SIREN",
    "SigmoidLayer",
    "SineLayer",
    "TanhLayer",
    "create_coords",
    "create_patches",
    "ff_embedding",
    "interpolate_image",
]

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Utils -----
def create_coords(size: int) -> torch.Tensor:
    """Creates a coordinate grid.

    Args:
        size: Size of the square coordinates grid.

    Returns:
        A ``torch.Tensor`` of shape :math:`(size, size, 2)` with normalized
        coords.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    return torch.from_numpy(coords).float()


def create_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Extracts patches into channels of a tensor.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in
            range :math:`[0, 1]`.
        kernel_size: Size of square patches. Default: ``1``.

    Returns:
        A ``torch.Tensor`` with patches in channels of shape :math:`(B, H', W', K^2)`.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must be a torch.Tensor of shape (B, C, H, W), got {image.shape}.")
    
    b, c, h, w = image.shape
    kernel     = torch.zeros(kernel_size ** 2, c, kernel_size, kernel_size, device=image.device)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i * kernel_size + j, :, i, j] = 1
    
    im_padded = nn.ReflectionPad2d(kernel_size // 2)(image)
    extracted = F.conv2d(im_padded, kernel, padding=0)
    return torch.movedim(extracted, 1, -1)


def interpolate_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """Resizes image to a new square resolution.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in
            range :math:`[0, 1]`.
        size: Target square size.

    Returns:
        A resized tensor as a ``torch.Tensor`` of shape :math:`(B, C, size, size)`.
    """
    return F.interpolate(image, size=(size, size), mode="bicubic")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Applies Fourier feature embedding to input tensor.

    Args:
        p: Input tensor to embed.
        B: Projection matrix as a ``torch.Tensor``. Default: ``None``.

    Returns:
        An embedded ``torch.Tensor`` with sine and cosine features.
    """
    if B is None:
        return p
    x_proj = (2 * np.pi * p) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ----- Linear Layer -----
class LinearLayer(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input)


class SigmoidLayer(nn.Module):
    r"""Applies an affine linear transformation with sigmoid activation to the
    incoming data: :math:`y = \sigma(xA^T + b)`, where :math:`\sigma` is the
    sigmoid function.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act    = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(input))


class TanhLayer(nn.Module):
    r"""Applies an affine linear transformation with tanh activation to the
    incoming data: :y = \tanh(xA^T + b), where :math:`\tanh` is the
    hyperbolic tangent function.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act    = nn.Tanh()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(input))


class ReLULayer(nn.Module):
    r"""Applies an affine linear transformation with ReLU activation to the
    incoming data: :y = \text{ReLU}(xA^T + b), where :math:`\text{ReLU}` is the
    rectified linear unit function.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act    = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(input))


class SineLayer(nn.Module):
    r"""Applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        is_first: First layer flag for weight initialization. Default: ``False``.
        w0: Frequency scaling factor. Default: ``30.0``.
        init_weights: Initializes weights if ``True``. Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.

    References:
        - Code: https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        w0          : float = 30.0,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.w0          = w0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            bound = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(input))

    def forward_with_intermediate(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = self.w0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class GaussLayer(nn.Module):
    r"""Applies an affine linear transformation with Gaussian activation to the
    incoming data: :math:`y = \exp(-(xA^T + b)^2)`, where :math:`\exp`
    is the exponential function.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        scale: Gaussian scale factor. Default: ``30.0``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        scale       : float = 30.0,
        *args, **kwargs
    ):
        super().__init__()
        self.scale  = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(self.scale * self.linear(input))**2)


class FINERLayer(nn.Module):
    r"""Applies an affine linear transformation with scaled sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b) \cdot \text{scale})`, where
    :math:`w_0` is a frequency factor and :math:`\sin` is the sine function.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        is_first: First layer flag for initialization. Default: ``False``.
        w0: Frequency scaling factor. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer. Default: ``20.0``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        bias            : bool  = True,
        is_first        : bool  = False,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.in_features      = in_features
        self.is_first         = is_first
        self.w0               = w0
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad   = scale_req_grad
        self.linear           = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        if self.first_bias_scale and self.is_first:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            bound = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def scale(self, liner: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor after linear transformation."""
        if self.scale_req_grad:
            return torch.abs(liner) + 1
        with torch.no_grad():
            return torch.abs(liner) + 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        linear = self.linear(input)
        scale  = self.scale(self.linear(input))
        return torch.sin(self.w0 * scale * linear)


class INRLayer(nn.Module):
    """Implements a general INR layer which automatically selects the nonlinearity.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        nonlinear: Nonlinearity type. Default: ``"sine"``.
        w0: Frequency scaling factor. Default: ``30.0``.
        is_first: First layer flag. Default: ``False``.
        is_last: Last layer flag, forces "sigmoid", as ``bool``. Default: ``False``.
    """
    
    INR_AF = Literal["sigmoid", "tanh", "relu", "sine", "gauss", "finer"]
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool   = True,
        nonlinear   : INR_AF = "sine",
        w0          : float  = 30.0,
        is_first    : bool   = False,
        is_last     : bool   = False,
        **kwargs
    ):
        super().__init__()
        if is_last:
            nonlinear = "sigmoid"
        
        layer_args = {
            "in_features" : in_features,
            "out_features": out_features,
            "bias"        : bias
        } | kwargs
        
        if nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(**layer_args)
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(**layer_args)
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(**layer_args)
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(**layer_args, is_first=is_first, w0=w0, init_weights=not is_last)
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(**layer_args)
        elif nonlinear == "finer":
            self.nonlinear = FINERLayer(**layer_args, is_first=is_first, w0=w0)
        else:
            raise ValueError(f"``nonlinear`` must be supported type, got {nonlinear}.")
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(input)
    

# ----- INR Network -----
class SIREN(nn.Module):
    """Implements the SIREN MLP module.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        first_w0: Frequency scaling factor for the first layer. Default: ``30.0``.
        hidden_w0: Frequency scaling factor for the hidden layers. Default: ``30.0``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        first_w0     : float = 30.0,
        hidden_w0    : float = 30.0,
        bias         : bool  = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(in_features, hidden_dim, bias, is_first=True, w0=first_w0),
            *[SineLayer(hidden_dim, hidden_dim, bias, is_first=False, w0=hidden_w0) for _ in range(hidden_layers)],
            LinearLayer(hidden_dim, out_features)
        )
        with torch.no_grad():
            self.net[-1].weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_w0,
                                          np.sqrt(6 / hidden_dim) / hidden_w0)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class GAUSS(nn.Module):
    """Implements the Gaussian MLP module.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        scale: Gaussian scale factor. Default: ``30.0``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        scale        : float = 30.0,
        bias         : bool  = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            GaussLayer(in_features, hidden_dim, bias, scale=scale),
            *[GaussLayer(hidden_dim, hidden_dim, bias, scale=scale) for _ in range(hidden_layers)],
            LinearLayer(hidden_dim, out_features)
        )
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
    

class FINER(nn.Module):
    """Implements the SIREN MLP module.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        first_w0: Frequency scaling factor for the first layer. Default: ``30.0``.
        hidden_w0: Frequency scaling factor for the hidden layers. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float`` or ``None``.
            Default: ``None``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        first_w0        : float = 30.0,
        hidden_w0       : float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False,
        bias            : bool  = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            FINERLayer(in_features, hidden_dim, bias, is_first=True, w0=first_w0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad),
            *[FINERLayer(hidden_dim, hidden_dim, bias, w0=hidden_w0, scale_req_grad=scale_req_grad) for _ in range(hidden_layers)],
            LinearLayer(hidden_dim, out_features)
        )
        with torch.no_grad():
            self.net[-1].weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_w0,
                                          np.sqrt(6 / hidden_dim) / hidden_w0)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
