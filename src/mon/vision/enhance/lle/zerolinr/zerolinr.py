#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Low-light Image Enhancement Network using Neural
Implicit Representations".
"""

__all__ = [
    "ZeroLINR",
]

from typing import Literal

import kornia
import numpy as np
import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t, functional as F
from mon.vision import filtering, types
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
INR_AF       = nn.inr_layer.INR_AF
MAPPING_FUNC = Literal["p", "v", "d", "e", "pv", "pd", "pe", "pvde"]


# ----- Utils -----
def create_coords(down_size: int) -> torch.Tensor:
    """Creates a coordinates grid.
    
    Args:
        down_size: The size of the coordinates grid.
    """
    h, w   = down_size, down_size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()
    return coords


def create_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information."""
    num_channels = types.image_num_channels(image)
    kernel       = torch.zeros((kernel_size ** 2, num_channels, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
    
    pad       = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


def interpolate_image(image: torch.Tensor, down_size: int) -> torch.Tensor:
    """Reshapes the image based on new resolution."""
    return F.interpolate(image, size=(down_size, down_size), mode="bicubic")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    if B is None:
        return p
    else:
        x_proj    = (2. * np.pi * p) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return embedding


def filter_up(x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Applies the guided filter to upscale the predicted image. """
    gf   = filtering.FastGuidedFilter(radius=radius)
    y_hr = gf(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0.0, 1.0)
    return y_hr


def replace_v_component(image_hsv: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
    image_hsv[:, -1, :, :] = v_new
    return image_hsv


def replace_i_component(image_hvi: torch.Tensor, i_new: torch.Tensor) -> torch.Tensor:
    """Replaces the `I` component of an HVI image `[1, 3, H, W]`."""
    image_hvi[:, 2, :, :] = i_new
    return image_hvi


def laplace(self, model_output: torch.Tensor, coords: torch.Tensor):
    y    = model_output
    x    = coords
    grad = self.gradient(y, x)
    return self.divergence(grad, x)


def divergence(model_output: torch.Tensor, coords: torch.Tensor):
    y   = model_output
    x   = coords
    div = 0.0
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i : i + 1]
    return div


def gradient(model_output: torch.Tensor, coords: torch.Tensor, grad_outputs: torch.Tensor = None):
    y = model_output
    x = coords
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
    

# ----- Loss -----
class Loss(nn.Loss):
    
    def __init__(
        self,
        loss_e_mean: float = 0.1,
        loss_w_f   : float = 1.0,
        loss_w_s   : float = 5.0,
        loss_w_e   : float = 8.0,
        loss_w_tv  : float = 20.0,
        loss_w_de  : float = 1.0,
        loss_w_c   : float = 5.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
        verbose    : bool = True,
    ):
        super().__init__(reduction=reduction)
        self.loss_w_f   = loss_w_f
        self.loss_w_s   = loss_w_s
        self.loss_w_e   = loss_w_e
        self.loss_w_tv  = loss_w_tv
        self.loss_w_de  = loss_w_de
        self.loss_w_c   = loss_w_c
        self.verbose    = verbose
        
        self.loss_e     = nn.ExposureValueControlLoss(16, loss_e_mean, reduction=reduction)
        self.loss_tv    = nn.TotalVariationLoss(reduction=reduction)
        self.loss_depth = nn.DepthAwareIlluminationLoss(reduction=reduction)
        self.loss_edge  = nn.EdgeAwareIlluminationLoss(reduction=reduction)
        self.loss_c     = nn.ColorConstancyLoss(reduction=reduction)
        
    def forward(
        self,
        enhanced: torch.Tensor,
        v_lr    : torch.Tensor,
        x_lr    : torch.Tensor,
        z_lr    : torch.Tensor,
        d_lr    : torch.Tensor = None,
        e_lr    : torch.Tensor = None,
    ) -> torch.Tensor:
        loss_f  = self.loss_w_f  * torch.mean(torch.abs(torch.pow(x_lr - v_lr, 2)))
        loss_s  = self.loss_w_s  * torch.mean(enhanced)
        loss_e  = self.loss_w_e  * torch.mean(self.loss_e(x_lr))
        loss_tv = self.loss_w_tv * self.loss_tv(x_lr)
        loss_c  = self.loss_w_c  * self.loss_c(enhanced)
        loss_de = 0.0
        if d_lr is not None:
            loss_de += self.loss_depth(x_lr, d_lr)
        if e_lr is not None:
            loss_de += self.loss_edge(x_lr, e_lr)
        loss_de = self.loss_w_de * loss_de
        loss    = loss_f + loss_s + loss_e + loss_tv + loss_de + loss_c
        
        if self.verbose:
            core.console.log(
                f"loss_f: {loss_f:.4f}, "
                f"loss_s: {loss_s:.4f}, "
                f"loss_e: {loss_e:.4f}, "
                f"loss_tv: {loss_tv:.4f}, "
                f"loss_de: {loss_de:.4f}, "
                f"loss_c: {loss_c:.4f}"
            )
        
        return loss


# ----- Module -----
class INF1_P(nn.Module):
    """Implicit Neural Function (INF) for 1-way residual reconstruction,
    i.e., f: (p) -> r.
    
    References:
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size      : int         = 1,
        down_size        : int         = 256,
        num_layers       : int         = 4,
        add_layers       : int         = 2,
        w0               : float       = 30.0,
        first_bias_scale : float       = 20.0,
        s_nonlinear      : INR_AF      = "finer",
        use_ff           : bool        = True,
        ff_gaussian_scale: float       = 10.0,
        v_nonlinear      : INR_AF      = "sine",
        reduce_channels  : bool        = False,
        weight_decay     : list[float] = [0.1, 0.0001, 0.001],
        *args, **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        
        # Construct MLP/INF
        hidden_dim = 256
        if use_ff:
            self.register_buffer("B1", torch.randn((hidden_dim, 2)) * ff_gaussian_scale)
            s_in_channels = hidden_dim * 2
        else:
            self.B1       = None
            s_in_channels = 2
        
        p_layers = [nn.INRLayer(s_in_channels, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            p_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        p_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        o_layers = [nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            o_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        o_layers.append(nn.INRLayer(hidden_dim, 1, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
        self.p_net = nn.Sequential(*p_layers)
        self.o_net = nn.Sequential(*o_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.p_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.o_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        p = ff_embedding(p, self.B1)
        return self.o_net(self.p_net(p))


class INF1_V(nn.Module):
    """Implicit Neural Function (INF) for 1-way residual reconstruction,
    i.e., f: (v) -> r.
    
    References:
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size      : int         = 1,
        down_size        : int         = 256,
        num_layers       : int         = 4,
        add_layers       : int         = 2,
        w0               : float       = 30.0,
        first_bias_scale : float       = 20.0,
        s_nonlinear      : INR_AF      = "finer",
        use_ff           : bool        = True,
        ff_gaussian_scale: float       = 10.0,
        v_nonlinear      : INR_AF      = "sine",
        reduce_channels  : bool        = False,
        weight_decay     : list[float] = [0.1, 0.0001, 0.001],
        *args, **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size

        # Construct MLP/INF
        patch_dim  = window_size ** 2
        hidden_dim = 256
        if use_ff:
            self.register_buffer("B2", torch.randn((hidden_dim, 2)) * ff_gaussian_scale)
            v_in_channels = patch_dim  * 2
        else:
            self.B2       = None
            v_in_channels = patch_dim
        
        v_layers = [nn.INRLayer(v_in_channels, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            v_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        v_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        o_layers = [nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            o_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        o_layers.append(nn.INRLayer(hidden_dim, 1, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
        self.v_net = nn.Sequential(*v_layers)
        self.o_net = nn.Sequential(*o_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.v_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.o_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_patch = create_patches(v, self.window_size)
        v_patch = ff_embedding(v_patch, self.B2)
        return self.o_net(self.v_net(v_patch))


class INF2(nn.Module):
    """Implicit Neural Function (INF) for 2-way residual reconstruction,
    i.e., f: (p,v) -> r.
    
    References:
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size      : int         = 1,
        down_size        : int         = 256,
        num_layers       : int         = 4,
        add_layers       : int         = 2,
        w0               : float       = 30.0,
        first_bias_scale : float       = 20.0,
        s_nonlinear      : INR_AF      = "finer",
        use_ff           : bool        = True,
        ff_gaussian_scale: float       = 10.0,
        v_nonlinear      : INR_AF      = "sine",
        reduce_channels  : bool        = False,
        weight_decay     : list[float] = [0.1, 0.0001, 0.001],
        *args, **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        
        # Construct MLP/INF
        patch_dim    = window_size ** 2
        hidden_dim   = 256
        mid_channels = hidden_dim // 2 if reduce_channels else hidden_dim
        if use_ff:
            self.register_buffer("B1", torch.randn((hidden_dim, 2)) * ff_gaussian_scale)
            s_in_channels = hidden_dim * 2
            self.register_buffer("B2", torch.randn((hidden_dim, patch_dim)) * ff_gaussian_scale)
            v_in_channels = hidden_dim * 2
        else:
            self.B1       = None
            self.B2       = None
            s_in_channels = 2
            v_in_channels = patch_dim
        
        p_layers = [nn.INRLayer(s_in_channels, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        v_layers = [nn.INRLayer(v_in_channels, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            p_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            v_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        p_layers.append(nn.INRLayer(hidden_dim, mid_channels, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        v_layers.append(nn.INRLayer(hidden_dim, mid_channels, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        o_layers = [nn.INRLayer(mid_channels * 2, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            o_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        o_layers.append(nn.INRLayer(hidden_dim, 1, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
        self.p_net = nn.Sequential(*p_layers)
        self.v_net = nn.Sequential(*v_layers)
        self.o_net = nn.Sequential(*o_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.p_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.v_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.o_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        p       = ff_embedding(p, self.B1)
        v_patch = create_patches(v, self.window_size)
        v_patch = ff_embedding(v_patch, self.B2)
        return self.o_net(torch.cat((self.p_net(p), self.v_net(v_patch)), -1))


class INF4(nn.Module):
    """Implicit Neural Function (INF) for 4-way residual reconstruction,
    i.e., f: (p,v,d,e) -> r.
    
    References:
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size      : int         = 1,
        down_size        : int         = 256,
        num_layers       : int         = 4,
        add_layers       : int         = 2,
        w0               : float       = 30.0,
        first_bias_scale : float       = 20.0,
        s_nonlinear      : INR_AF      = "finer",
        use_ff           : bool        = True,
        ff_gaussian_scale: float       = 10.0,
        v_nonlinear      : INR_AF      = "sine",
        reduce_channels  : bool        = False,
        weight_decay     : list[float] = [0.1, 0.0001, 0.001],
        *args, **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        
        # Construct MLP/INF
        patch_dim    = window_size ** 2
        hidden_dim   = 256
        mid_channels = hidden_dim // 4 if reduce_channels else hidden_dim
        if use_ff:
            self.register_buffer("B1", torch.randn((hidden_dim, 2)) * ff_gaussian_scale)
            s_in_channels = hidden_dim * 2
            self.register_buffer("B2", torch.randn((hidden_dim, patch_dim)) * ff_gaussian_scale)
            v_in_channels = hidden_dim * 2
        else:
            self.B1       = None
            self.B2       = None
            s_in_channels = 2
            v_in_channels = patch_dim
        
        p_layers = [nn.INRLayer(s_in_channels, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        v_layers = [nn.INRLayer(v_in_channels, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        d_layers = [nn.INRLayer(v_in_channels, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        e_layers = [nn.INRLayer(v_in_channels, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            p_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            v_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            d_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            e_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        p_layers.append(nn.INRLayer(hidden_dim, mid_channels, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        v_layers.append(nn.INRLayer(hidden_dim, mid_channels, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        d_layers.append(nn.INRLayer(hidden_dim, mid_channels, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        e_layers.append(nn.INRLayer(hidden_dim, mid_channels, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        o_layers = [nn.INRLayer(mid_channels * 4, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            o_layers.append(nn.INRLayer(hidden_dim, hidden_dim, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        o_layers.append(nn.INRLayer(hidden_dim, 1, v_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
        self.p_net = nn.Sequential(*p_layers)
        self.v_net = nn.Sequential(*v_layers)
        self.d_net = nn.Sequential(*d_layers)
        self.e_net = nn.Sequential(*e_layers)
        self.o_net = nn.Sequential(*o_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.p_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.v_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.d_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.e_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.o_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, p: torch.Tensor, v: torch.Tensor, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        p       = ff_embedding(p, self.B1)
        v_patch = create_patches(v, self.window_size)
        d_patch = create_patches(d, self.window_size)
        e_patch = create_patches(e, self.window_size)
        v_patch = ff_embedding(v_patch, self.B2)
        d_patch = ff_embedding(d_patch, self.B2)
        e_patch = ff_embedding(e_patch, self.B2)
        return self.o_net(torch.cat((self.p_net(p), self.v_net(v_patch), self.d_net(d_patch), self.e_net(e_patch)), -1))


# ----- Model -----
@MODELS.register(name="zerolinr", arch="zerolinr")
class ZeroLINR(base.ImageEnhancementModel):
    """Zero-LINR model for low-light image enhancement."""
    
    arch     : str          = "zerolinr"
    name     : str          = "zerolinr"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        mapping_func     : MAPPING_FUNC = "pvde",
        window_size      : int          = 9,
        down_size        : int          = 256,
        num_layers       : int          = 4,
        add_layers       : int          = 2,
        w0               : float        = 30.0,
        first_bias_scale : float        = 20.0,
        s_nonlinear      : INR_AF       = "finer",
        use_ff           : bool         = True,
        ff_gaussian_scale: float        = 10.0,
        v_nonlinear      : INR_AF       = "finer",
        reduce_channels  : bool         = False,
        depth_threshold  : float        = 1.0,
        edge_threshold   : float        = 0.05,
        # Post-process
        gf_radius        : int          = 7,
        use_denoise      : bool         = False,
        denoise_ksize    : _size_2_t    = (3, 3),
        denoise_color    : float        = 0.5,
        denoise_space    : _size_2_t    = (1.5, 1.5),
        iters            : int          = 100,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mapping_func    = mapping_func
        self.down_size       = down_size
        self.depth_threshold = depth_threshold
        self.edge_threshold  = edge_threshold
        self.gf_radius       = gf_radius
        self.use_denoise     = use_denoise
        self.denoise_ksize   = tuple(denoise_ksize)
        self.denoise_color   = denoise_color
        self.denoise_space   = tuple(denoise_space)
        self.iters           = iters
        weight_decay         = [0.1, 0.0001, 0.001]
        
        # Model
        loss_w_de = self.loss["loss_w_de"]
        if mapping_func in ["p"]:
            inf       = INF1_P
            loss_w_de = 0.0
        elif mapping_func in ["v"]:
            inf       = INF1_V
            loss_w_de = 0.0
        elif mapping_func in ["d"]:
            inf       = INF1_V
        elif mapping_func in ["e"]:
            inf       = INF1_V
            loss_w_de = 0.0
        elif mapping_func in ["pv"]:
            inf       = INF2
            loss_w_de = 0.0
        elif mapping_func in ["pd"]:
            inf       = INF2
        elif mapping_func in ["pe"]:
            inf       = INF2
            loss_w_de = 0.0
        elif mapping_func in ["pvde"]:
            inf       = INF4
        else:
            raise ValueError(f"[mapping_func] must be one of {MAPPING_FUNC}, got {mapping_func}.")
        self.inf = inf(
            window_size       = window_size,
            down_size         = down_size,
            num_layers        = num_layers,
            add_layers        = add_layers,
            w0                = w0,
            first_bias_scale  = first_bias_scale,
            s_nonlinear       = s_nonlinear,
            use_ff            = use_ff,
            ff_gaussian_scale = ff_gaussian_scale,
            v_nonlinear       = v_nonlinear,
            reduce_channels   = reduce_channels,
            weight_decay      = weight_decay,
        )
        
        # Optimizer
        self.configure_optimizers()
        
        # Loss
        self.loss = Loss(
            loss_e_mean = self.loss["loss_e_mean"],
            loss_w_f    = self.loss["loss_w_f"],
            loss_w_s    = self.loss["loss_w_s"],
            loss_w_e    = self.loss["loss_w_e"],
            loss_w_tv   = self.loss["loss_w_tv"],
            loss_w_de   = loss_w_de,
            loss_w_c    = self.loss["loss_w_c"],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    def compute_efficiency_score(self, image_size: _size_2_t = 512) -> tuple[float, float]:
        """Compute model efficiency score (FLOPs, params).

        Args:
            image_size: Input size as ``int`` or [H, W]. Default is ``512``.

        Returns:
            Tuple of (FLOPs, parameter count) as ``float`` values.
        """
        from fvcore.nn import parameter_count
        
        h, w      = types.image_size(image_size)
        datapoint = {
            "image": torch.rand(1, 3, h, w).to(self.device),
            "depth": torch.rand(1, 1, h, w).to(self.device)
        }

        flops, params = core.thop.custom_profile(self, inputs=datapoint, verbose=False)
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params

        return flops, params
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs  = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        v_lr     = datapoint["v_lr"]
        d_lr     = datapoint["d_lr"]
        e_lr     = datapoint["e_lr"]
        x_lr     = outputs["x_lr"]
        z_lr     = outputs["z_lr"]
        enhanced = outputs["enhanced"]
        loss     = self.loss(enhanced, v_lr, x_lr, z_lr, d_lr, e_lr)
        
        return outputs | {
            "loss": loss,
        }
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        # Input
        if not core.are_all_items_in_dict(
            ["p", "v", "d", "e", "v_lr", "d_lr", "e_lr"],
            datapoint
        ):
            datapoint = self.prepare_input(datapoint)
        hsv  = datapoint["hsv"].clone()
        p    = datapoint["p"]
        v    = datapoint["v"]
        v_lr = datapoint["v_lr"]
        d_lr = datapoint["d_lr"]
        e_lr = datapoint["e_lr"]

        # Mapping
        if self.mapping_func in ["p", "v", "pv"]:
            r = self.inf(p, v_lr)
        elif self.mapping_func in ["d", "pd"]:
            r = self.inf(p, d_lr)
        elif self.mapping_func in ["e", "pe"]:
            r = self.inf(p, e_lr)
        elif self.mapping_func in ["pvde"]:
            r = self.inf(p, v_lr, d_lr, e_lr)
        else:
            raise ValueError(f"[mapping_func] must be one of {MAPPING_FUNC}, got {self.mapping}.")
        r_lr = r.view(1, 1, self.down_size, self.down_size)
        
        # Enhance
        if self.depth_threshold > 0:
            r_lr = r_lr * (1 + self.depth_threshold * (1 - d_lr / d_lr.max()))
        x_lr = v_lr + r_lr
        z_lr = v_lr / (x_lr + 1e-8)
        
        # Post-process
        if self.use_denoise:
            z_lr = kornia.filters.bilateral_blur(z_lr, self.denoise_ksize, self.denoise_color, self.denoise_space)
        z   = filter_up(v_lr, z_lr, v, self.gf_radius)
        hsv = replace_v_component(hsv, z)
        rgb = kornia.color.hsv_to_rgb(hsv)

        return {
            "r_lr"    : r_lr,
            "x_lr"    : x_lr,
            "z_lr"    : z_lr,
            "enhanced": rgb,
        }
    
    def prepare_input(self, datapoint: dict) -> dict:
        """Prepares input for the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of prepared inputs.
        """
        image = datapoint["image"]
        hsv   = kornia.color.rgb_to_hsv(image)
        p     = create_coords(self.down_size).to(image.device)
        v     = kornia.color.rgb_to_hsv(image)[:, 2:3, :, :]
        d     = datapoint.get("depth", None)
        e     = types.boundary_aware_prior(d, self.edge_threshold) if d is not None else None
        v_lr  = interpolate_image(v, self.down_size)
        d_lr  = interpolate_image(d, self.down_size) if d is not None else None
        e_lr  = interpolate_image(e, self.down_size) if e is not None else None
        return datapoint | {
            "hsv" : hsv,
            "p"   : p,
            "v"   : v,
            "d"   : d,
            "e"   : e,
            "v_lr": v_lr,
            "d_lr": d_lr,
            "e_lr": e_lr,
        }
    
    # ----- Predict -----
    def infer(self, datapoint: dict, reset_weights: bool = True, *args, **kwargs) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            reset_weights: Whether to reset the weights before training. Default is ``True``.
            
        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Initialize training components
        if reset_weights:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer = self.optimizer.get("optimizer", None)
        optimizer = optimizer or nn.Adam(self.parameters(), lr=0.00001, weight_decay=0.0003)

        # Input
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        datapoint = self.prepare_input(datapoint)
        
        # Optimize
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            if loss is not None:
                loss.backward(retain_graph=True)
                optimizer.step()
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        
        # Return
        return outputs | {
            "time": timer.avg_time,
        }
