#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-MIE-Multiscale.

This module implement our idea: "Zero-shot Multimodal Illumination Estimation
for Low-light Image Enhancement via Neural Implicit Representations".
"""

from __future__ import annotations

__all__ = [
    "ZeroMIE_MS",
]

from abc import ABC
from typing import Literal

import kornia
import torch
from fvcore.nn import parameter_count
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.models import vgg

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base, utils

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
LDA          = nn.LayeredFeatureAggregation


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_mean    : float = 0.6,
        exp_weight  : float = 10.0,
        spa_weight  : float = 1.0,
        color_weight: float = 5.0,
        tv_weight   : float = 1600,
        depth_weight: float = 1.0,
        edge_weight : float = 1.0,
        reduction   : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.exp_weight   = exp_weight
        self.spa_weight   = spa_weight
        self.color_weight = color_weight
        self.tv_weight    = tv_weight
        self.depth_weight = depth_weight
        self.edge_weight  = edge_weight
        
        self.exp_loss   = nn.ExposureControlLoss(16, exp_mean, reduction=reduction)
        self.spa_loss   = nn.SpatialConsistencyLoss(8, reduction=reduction)
        self.color_loss = nn.ColorConstancyLoss(reduction=reduction)
        self.tv_loss    = nn.TotalVariationLoss(reduction=reduction)
        self.depth_loss = nn.MultiscaleDepthConsistencyLoss(reduction=reduction)
        self.edge_loss  = nn.EdgeAwareDepthConsistencyLoss(reduction=reduction)
        
    def forward(
        self,
        image          : torch.Tensor,
        image_lr       : torch.Tensor,
        illumination_lr: torch.Tensor,
        enhanced       : torch.Tensor,
        enhanced_lr    : torch.Tensor,
        depth_lr       : torch.Tensor = None,
    ) -> torch.Tensor:
        exp_loss   = self.exp_weight   * self.exp_loss(input=enhanced)
        spa_loss   = self.spa_weight   * self.spa_loss(input=enhanced, target=image)
        color_loss = self.color_weight * self.color_loss(input=enhanced)
        tv_loss    = self.tv_weight    * self.tv_loss(input=illumination_lr)
        if depth_lr is not None:
            depth_loss = self.depth_weight * self.depth_loss(enhanced_lr, depth_lr)
            edge_loss  = self.edge_weight  *  self.edge_loss(enhanced_lr, depth_lr)
        else:
            depth_loss = 0
            edge_loss  = 0
        loss = exp_loss + spa_loss + color_loss + tv_loss + depth_loss + edge_loss
        '''
        print(
            f"exp_loss: {exp_loss:.4f}, "
            f"spa_loss: {spa_loss:.4f}, "
            f"color_loss: {color_loss:.4f}, "
            f"tv_loss: {tv_loss:.4f}, "
            f"depth_loss: {depth_loss:.4f}, "
            f"edge_loss: {edge_loss:.4f}"
        )
        '''
        return loss


class LossHSV(nn.Loss):
    
    def __init__(
        self,
        exp_mean    : float = 0.1,
        exp_weight  : float = 8.0,
        spa_weight  : float = 1.0,
        tv_weight   : float = 20.0,
        spar_weight : float = 5.0,
        depth_weight: float = 0.0,
        edge_weight : float = 0.0,
        color_weight: float = 0.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.exp_weight   = exp_weight
        self.spa_weight   = spa_weight
        self.tv_weight    = tv_weight
        self.spar_weight  = spar_weight
        self.depth_weight = depth_weight
        self.edge_weight  = edge_weight
        self.color_weight = color_weight
        
        self.exp_loss   = nn.ExposureValueControlLoss(16, exp_mean, reduction=reduction)
        self.tv_loss    = nn.TotalVariationLoss(reduction=reduction)
        self.depth_loss = nn.MultiscaleDepthConsistencyLoss(reduction=reduction)
        self.edge_loss  = nn.EdgeAwareDepthConsistencyLoss(reduction=reduction)
        self.color_loss = nn.ColorConstancyLoss(reduction=reduction)
        
    def forward(
        self,
        image          : torch.Tensor,
        image_lr       : torch.Tensor,
        illumination_lr: torch.Tensor,
        enhanced       : torch.Tensor,
        enhanced_lr    : torch.Tensor,
        depth          : torch.Tensor = None,
    ) -> torch.Tensor:
        exp_loss   = self.exp_weight   * torch.mean(self.exp_loss(illumination_lr))
        spa_loss   = self.spa_weight   * torch.mean(torch.abs(torch.pow(illumination_lr - image_lr, 2)))
        tv_loss    = self.tv_weight    * self.tv_loss(illumination_lr)
        spar_loss  = self.spar_weight  * torch.mean(enhanced)
        color_loss = self.color_weight * self.color_loss(enhanced)
        # noise_loss = ((enhanced - image) ** 2).mean()
        if depth is not None:
            depth_loss = self.depth_weight * (self.depth_loss(enhanced_lr, depth))
            edge_loss  = self.edge_weight  * ( self.edge_loss(enhanced_lr, depth))
        else:
            depth_loss = 0
            edge_loss  = 0
        
        loss = exp_loss + spa_loss + tv_loss + spar_loss + color_loss + depth_loss + edge_loss  # + noise_loss
        '''
        print(
            f"exp_loss: {exp_loss:.4f}, "
            f"spa_loss: {spa_loss:.4f}, "
            f"tv_loss: {tv_loss:.4f}, "
            f"spar_loss: {spar_loss:.4f}, "
            f"color_loss: {color_loss:.4f}, "
            f"depth_loss: {depth_loss:.4f}, "
            f"edge_loss: {edge_loss:.4f}"
        )
        '''
        return loss


class LossS(nn.Loss):
    """
    λ1, λ2, λ3, λ4 = {0.40, 0.05, 0.15, 0.40} for the LOL dataset.
    λ1, λ2, λ3, λ4 = {0.35, 0.10, 0.25, 0.30} for the VE-LOL dataset.
    """
    
    def __init__(
        self,
        str_weight: float = 0.40,
        tv_weight : float = 0.05,
        reg_weight: float = 0.15,
        per_weight: float = 0.40,
        reduction : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.str_weight = str_weight
        self.tv_weight  = tv_weight
        self.reg_weight = reg_weight
        self.per_weight = per_weight
        
        self.ms_ssim_loss = nn.MSSSIMLoss(data_range=1.0)
        self.ssim_loss    = nn.SSIMLoss(data_range=1.0, non_negative_ssim=True)
        self.per_loss     = nn.PerceptualLoss(
            net        = vgg.vgg19(weights=vgg.VGG19_Weights.IMAGENET1K_V1).features,
            layers     = ["26"],
            preprocess = True,
            reduction  = reduction,
        )
        self.tv_loss = nn.TotalVariationLoss(reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        str_loss = self.ms_ssim_loss(input, target) + self.ssim_loss(input, target)
        per_loss = self.per_loss(input, target)
        reg_loss = self.region_loss(input, target)
        tv_loss  = self.tv_loss(input)
        loss     = (
              self.str_weight * str_loss
            + self.tv_weight  * tv_loss
            + self.reg_weight * reg_loss
            + self.per_weight * per_loss
        )
        return loss
    
    @staticmethod
    def region_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gray     = (
              0.30 * target[:, 0, :, :]
            + 0.59 * target[:, 1, :, :]
            + 0.11 * target[:, 2, :, :]
        )
        gray     = gray.view(-1)
        value    = -torch.topk(-gray, int(gray.shape[0] * 0.4))[0][0]
        weight   = 1 * (target > value) + 4 * (target <= value)
        abs_diff = torch.abs(input - target)
        return torch.mean(weight * abs_diff)

# endregion


# region Modules

class FiLM(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        # Two linear layers to generate scale and shift from depth information
        self.fc_scale = nn.Linear(1, in_channels)  # Scale from depth
        self.fc_shift = nn.Linear(1, in_channels)  # Shift from depth

    def forward(self, x, depth_map):
        batch_size = depth_map.size(0)  # This should be 1 for your case
        height, width, _ = x.size()  # height=256, width=256, in_channels=64

        # Flatten depth_map to [B, H*W, 1]
        depth_map_flat = depth_map.view(batch_size, -1, 1)  # Shape: [1, 256*256, 1]

        # Calculate scale and shift based on depth
        scale = self.fc_scale(depth_map_flat)  # Shape: [1, H*W, in_channels]
        shift = self.fc_shift(depth_map_flat)  # Shape: [1, H*W, in_channels]

        # Reshape x to [B, H*W, in_channels]
        x = x.view(-1, x.size(2))  # Flatten x to shape [H*W, in_channels] => [65536, 64]

        # Apply FiLM to input features
        x = (x * scale.view(-1, x.size(1))) + shift.view(-1, x.size(1))  # Apply modulation

        # Reshape back to original dimensions [H, W, C]
        return x.view(height, width, -1)  # Shape: [H, W, in_channels]


class CrossAttentionLayer(nn.Module):
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query   = query.permute(1, 0, 2)  # [seq_len, batch_size, dim]
        key     =   key.permute(1, 0, 2)
        value   = value.permute(1, 0, 2)
        attn, _ = self.attn(query, key, value)
        return attn.permute(1, 0, 2)  # Back to [batch_size, seq_len, dim]


class MLP(nn.Module, ABC):
    
    def interpolate_image(self, image: torch.Tensor, down_size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
    
    @staticmethod
    def filter_up(
        x_lr  : torch.Tensor,
        y_lr  : torch.Tensor,
        x_hr  : torch.Tensor,
        radius: int = 3
    ):
        """Applies the guided filter to upscale the predicted image. """
        gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
    
    @staticmethod
    def replace_v_component(image_hsv: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
    @staticmethod
    def replace_i_component(image_hvi: torch.Tensor, i_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `I` component of an HVI image `[1, 3, H, W]`."""
        image_hvi[:, 2, :, :] = i_new
        return image_hvi


class MLP_RGB(MLP):
    
    def __init__(
        self,
        window_size     : list[int]   = [3, 7],
        hidden_channels : list[int]   = [128, 256],
        down_size       : list[int]   = [128, 256],
        hidden_layers   : list[int]   = [2, 2],
        out_layers      : list[int]   = [1, 1],
        omega_0         : list[float] = [30.0, 30.0],
        first_bias_scale: list[float] = [None, None],
        nonlinear       : list[Literal["finer", "gauss", "relu", "sigmoid", "sine"]] = ["sine", "sine"],
        weight_decay    : list[float] = [0.1, 0.0001, 0.001],
        dba_eps         : float       = 0.05,
        gf_radius       : int         = 3,
        denoise         : bool        = False,
        denoise_ksize   : list[float] = (3, 3),
        denoise_color   : float       = 0.5,
        denoise_space   : list[float] = (1.5, 1.5),
        *args, **kwargs
    ):
        super().__init__()
        self.window_size   = window_size
        self.down_size     = down_size
        self.gf_radius     = gf_radius
        self.denoise       = denoise
        self.denoise_ksize = denoise_ksize
        self.denoise_color = denoise_color
        self.denoise_space = denoise_space
        self.out_channels  = 3
        self.num_scales    = len(window_size)
        
        self.value_nets  = nn.ModuleList()
        self.coords_nets = nn.ModuleList()
        self.output_nets = nn.ModuleList()
        for i in range(self.num_scales):
            window_size_      = window_size[i]
            hidden_channels_  = hidden_channels[i]
            down_size_        = down_size[i]
            hidden_layers_    = hidden_layers[i]
            out_layers_       = out_layers[i]
            omega_0_          = omega_0[i]
            first_bias_scale_ = first_bias_scale[i]
            nonlinear_        = nonlinear[i]
            mid_channels_     = hidden_channels_ // 2
            self.value_nets.append(nn.ContextImplicitFeatureEncoder(window_size_, mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[1]))
            self.coords_nets.append(nn.ContextImplicitCoordinatesEncoder(mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[0]))
            self.output_nets.append(nn.ContextImplicitDecoder(hidden_channels_, self.out_channels, out_layers_, omega_0_, nonlinear_, weight_decay[2]))
        self.dba = nn.BoundaryAwarePrior(eps=dba_eps, normalized=False)
        self.w_0 = nn.Parameter(torch.Tensor([0.5]))
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Multi-scale
        outputs = {
            "image": image,
            "depth": depth,
            "edge" : edge,
        }
        enhanced_list = []
        for i in range(self.num_scales):
            # Mapping
            image_lr_i, value_inr_i = self.value_nets[i](image)
            depth_lr_i              = self.interpolate_image(depth, self.down_size[i])
            edge_lr_i               = self.interpolate_image(edge,  self.down_size[i])
            coords_i                = self.coords_nets[i](image)
            # Combining
            illu_res_lr_i = self.output_nets[i](torch.cat([value_inr_i, coords_i], -1))
            illu_res_lr_i = illu_res_lr_i.view(1, self.out_channels, self.down_size[i], self.down_size[i])
            # Enhancement
            illu_lr_i     = illu_res_lr_i + image_lr_i
            enhanced_lr_i = image_lr_i / (illu_lr_i + 1e-8)
            if self.denoise:
                enhanced_lr_i = kornia.filters.bilateral_blur(enhanced_lr_i, self.denoise_ksize, self.denoise_color, self.denoise_space)
            enhanced_i = self.filter_up(image_lr_i, enhanced_lr_i, image, self.gf_radius)
            enhanced_i = enhanced_i / torch.max(enhanced_i)
            # Save
            enhanced_list.append(enhanced_i)
            outputs |= {
                f"image_lr_{i}"   : image_lr_i,
                f"depth_lr_{i}"   : depth_lr_i,
                f"edge_lr_{i}"    : edge_lr_i,
                f"illu_res_lr_{i}": illu_res_lr_i,
                f"illu_lr_{i}"    : illu_lr_i,
                f"enhanced_lr_{i}": enhanced_lr_i,
                f"enhanced_{i}"   : enhanced_i,
            }
        # Combine Enhanced
        # enhanced  = sum(enhanced_list) / self.num_scales
        enhanced  = enhanced_list[0] * self.w_0 + enhanced_list[1] * (1 - self.w_0)
        outputs  |= {"enhanced": enhanced}
        # Return
        return outputs


class MLP_RGB_D(MLP):
    
    def __init__(
        self,
        window_size     : list[int]   = [3, 7],
        hidden_channels : list[int]   = [128, 256],
        down_size       : list[int]   = [128, 256],
        hidden_layers   : list[int]   = [2, 2],
        out_layers      : list[int]   = [1, 1],
        omega_0         : list[float] = [30.0, 30.0],
        first_bias_scale: list[float] = [None, None],
        nonlinear       : list[Literal["finer", "gauss", "relu", "sigmoid", "sine"]] = ["sine", "sine"],
        weight_decay    : list[float] = [0.1, 0.0001, 0.001],
        dba_eps         : float       = 0.05,
        gf_radius       : int         = 3,
        denoise         : bool        = False,
        denoise_ksize   : list[float] = (3, 3),
        denoise_color   : float       = 0.5,
        denoise_space   : list[float] = (1.5, 1.5),
        *args, **kwargs
    ):
        super().__init__()
        self.window_size   = window_size
        self.down_size     = down_size
        self.gf_radius     = gf_radius
        self.denoise       = denoise
        self.denoise_ksize = denoise_ksize
        self.denoise_color = denoise_color
        self.denoise_space = denoise_space
        self.out_channels  = 3
        self.num_scales    = len(window_size)
        
        self.value_nets  = nn.ModuleList()
        self.depth_nets  = nn.ModuleList()
        self.edge_nets   = nn.ModuleList()
        self.coords_nets = nn.ModuleList()
        self.output_nets = nn.ModuleList()
        for i in range(self.num_scales):
            window_size_      = window_size[i]
            hidden_channels_  = hidden_channels[i]
            down_size_        = down_size[i]
            hidden_layers_    = hidden_layers[i]
            out_layers_       = out_layers[i]
            omega_0_          = omega_0[i]
            first_bias_scale_ = first_bias_scale[i]
            nonlinear_        = nonlinear[i]
            mid_channels_     = hidden_channels_ // 4
            self.value_nets.append(nn.ContextImplicitFeatureEncoder(window_size_, mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[1]))
            self.depth_nets.append(nn.ContextImplicitFeatureEncoder(window_size_, mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[1]))
            self.edge_nets.append(nn.ContextImplicitFeatureEncoder(window_size_, mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[1]))
            self.coords_nets.append(nn.ContextImplicitCoordinatesEncoder(mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[0]))
            self.output_nets.append(nn.ContextImplicitDecoder(hidden_channels_, self.out_channels, out_layers_, omega_0_, nonlinear_, weight_decay[2]))
        self.dba = nn.BoundaryAwarePrior(eps=dba_eps, normalized=False)
        self.w_0 = nn.Parameter(torch.Tensor([0.5]))
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Multi-scale
        outputs = {
            "image": image,
            "depth": depth,
            "edge" : edge,
        }
        enhanced_list = []
        for i in range(self.num_scales):
            # Mapping
            image_lr_i, value_inr_i = self.value_nets[i](image)
            depth_lr_i, depth_inr_i = self.depth_nets[i](depth)
            edge_lr_i,  edge_inr_i  = self.edge_nets[i](edge)
            coords_i                = self.coords_nets[i](image)
            # Combining
            illu_res_lr_i = self.output_nets[i](torch.cat([value_inr_i, depth_inr_i, edge_inr_i, coords_i], -1))
            illu_res_lr_i = illu_res_lr_i.view(1, self.out_channels, self.down_size[i], self.down_size[i])
            # Enhancement
            illu_lr_i     = illu_res_lr_i + image_lr_i
            enhanced_lr_i = image_lr_i / (illu_lr_i + 1e-8)
            if self.denoise:
                enhanced_lr_i = kornia.filters.bilateral_blur(enhanced_lr_i, self.denoise_ksize, self.denoise_color, self.denoise_space)
            enhanced_i = self.filter_up(image_lr_i, enhanced_lr_i, image, self.gf_radius)
            enhanced_i = enhanced_i / torch.max(enhanced_i)
            # Save
            enhanced_list.append(enhanced_i)
            outputs |= {
                f"image_lr_{i}"   : image_lr_i,
                f"depth_lr_{i}"   : depth_lr_i,
                f"edge_lr_{i}"    : edge_lr_i,
                f"illu_res_lr_{i}": illu_res_lr_i,
                f"illu_lr_{i}"    : illu_lr_i,
                f"enhanced_lr_{i}": enhanced_lr_i,
                f"enhanced_{i}"   : enhanced_i,
            }
        # Combine Enhanced
        # enhanced  = sum(enhanced_list) / self.num_scales
        enhanced  = enhanced_list[0] * self.w_0 + enhanced_list[1] * (1 - self.w_0)
        outputs  |= {"enhanced": enhanced}
        # Return
        return outputs


class MLP_HSV_V(MLP):
    
    def __init__(
        self,
        window_size     : list[int]   = [3, 7],
        hidden_channels : list[int]   = [128, 256],
        down_size       : list[int]   = [128, 256],
        hidden_layers   : list[int]   = [2, 2],
        out_layers      : list[int]   = [1, 1],
        omega_0         : list[float] = [30.0, 30.0],
        first_bias_scale: list[float] = [None, None],
        nonlinear       : list[Literal["finer", "gauss", "relu", "sigmoid", "sine"]] = ["sine", "sine"],
        weight_decay    : list[float] = [0.1, 0.0001, 0.001],
        dba_eps         : float       = 0.05,
        gf_radius       : int         = 3,
        denoise         : bool        = False,
        denoise_ksize   : list[float] = (3, 3),
        denoise_color   : float       = 0.5,
        denoise_space   : list[float] = (1.5, 1.5),
        *args, **kwargs
    ):
        super().__init__()
        self.window_size   = window_size
        self.down_size     = down_size
        self.gf_radius     = gf_radius
        self.denoise       = denoise
        self.denoise_ksize = denoise_ksize
        self.denoise_color = denoise_color
        self.denoise_space = denoise_space
        self.out_channels  = 1
        self.num_scales    = len(window_size)
        
        self.value_nets   = nn.ModuleList()
        self.coords_nets  = nn.ModuleList()
        self.output_nets  = nn.ModuleList()
        for i in range(self.num_scales):
            window_size_      = window_size[i]
            hidden_channels_  = hidden_channels[i]
            down_size_        = down_size[i]
            hidden_layers_    = hidden_layers[i]
            out_layers_       = out_layers[i]
            omega_0_          = omega_0[i]
            first_bias_scale_ = first_bias_scale[i]
            nonlinear_        = nonlinear[i]
            mid_channels_     = hidden_channels_ // 2
            self.value_nets.append(nn.ContextImplicitFeatureEncoder(window_size_, mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[1]))
            self.coords_nets.append(nn.ContextImplicitCoordinatesEncoder(mid_channels_, down_size_, hidden_layers_, omega_0_, first_bias_scale_, nonlinear_, weight_decay[0]))
            self.output_nets.append(nn.ContextImplicitDecoder(hidden_channels_, self.out_channels, out_layers_, omega_0_, nonlinear_, weight_decay[2]))
        self.dba = nn.BoundaryAwarePrior(eps=dba_eps, normalized=False)
        self.w_0 = nn.Parameter(torch.Tensor([0.5]))
        # self.lfa = nn.LayeredFeatureAggregation([3] * self.num_scales, 3)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge      = self.dba(depth)
        image_hsv = core.rgb_to_hsv(image)
        image_v   = core.rgb_to_v(image)
        # Multi-scale
        outputs = {
            "image": image,
            "depth": depth,
            "edge" : edge,
        }
        enhanced_list = []
        for i in range(self.num_scales):
            # Mapping
            image_lr_i, value_inr_i = self.value_nets[i](image_v)
            depth_lr_i              = self.interpolate_image(depth, self.down_size[i])
            edge_lr_i               = self.interpolate_image(edge,  self.down_size[i])
            coords_i                = self.coords_nets[i](image_v)
            # Combining
            illu_res_lr_i = self.output_nets[i](torch.cat([value_inr_i, coords_i], -1))
            illu_res_lr_i = illu_res_lr_i.view(1, self.out_channels, self.down_size[i], self.down_size[i])
            # Enhancement
            illu_lr_i     = illu_res_lr_i + image_lr_i
            enhanced_lr_i = image_lr_i / (illu_lr_i + 1e-8)
            if self.denoise:
                enhanced_lr_i = kornia.filters.bilateral_blur(enhanced_lr_i, self.denoise_ksize, self.denoise_color, self.denoise_space)
            enhanced_v_i = self.filter_up(image_lr_i, enhanced_lr_i, image_v, self.gf_radius)
            enhanced_i   = self.replace_v_component(image_hsv, enhanced_v_i)
            enhanced_i   = core.hsv_to_rgb(enhanced_i.clone())
            enhanced_i   = enhanced_i / torch.max(enhanced_i)
            # Save
            enhanced_list.append(enhanced_i)
            outputs |= {
                f"image_lr_{i}"   : image_lr_i,
                f"depth_lr_{i}"   : depth_lr_i,
                f"edge_lr_{i}"    : edge_lr_i,
                f"illu_res_lr_{i}": illu_res_lr_i,
                f"illu_lr_{i}"    : illu_lr_i,
                f"enhanced_lr_{i}": enhanced_lr_i,
                f"enhanced_{i}"   : enhanced_i,
            }
        # Combine Enhanced
        enhanced  = sum(enhanced_list) / self.num_scales
        # enhanced  = enhanced_list[0] * self.w_0 + enhanced_list[1] * (1 - self.w_0)
        # enhanced  = self.lfa(enhanced_list)
        outputs  |= {"enhanced": enhanced}
        # Return
        return outputs

# endregion


# region Model

@MODELS.register(name="zero_mie_ms", arch="zero_mie")
class ZeroMIE_MS(base.ImageEnhancementModel):
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_mie"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name            : str         = "zero_mie_ms",
        color_space     : Literal["rgb", "rgb_d", "hsv_v"] = "hsv_v",
        window_size     : list[int]   = [7, 7],
        hidden_channels : list[int]   = [128, 256],
        down_size       : list[int]   = [128, 256],
        hidden_layers   : list[int]   = [2, 2],
        out_layers      : list[int]   = [1, 1],
        omega_0         : list[float] = [30.0, 30.0],
        first_bias_scale: list[float] = [None, None],
        nonlinear       : list[Literal["finer", "gauss", "relu", "sigmoid", "sine"]] = ["sine", "sine"],
        dba_eps         : float       = 0.05,
        gf_radius       : int         = 3,
        denoise         : bool        = False,
        denoise_ksize   : list[float] = (3, 3),
        denoise_color   : float       = 0.5,
        denoise_space   : list[float] = (1.5, 1.5),
        # Loss
        loss_hsv        : bool        = True,
        exp_mean        : float       = 0.6,
        exp_weight      : float       = 10,
        spa_weight      : float       = 1,
        color_weight    : float       = 5,
        tv_weight       : float       = 1600,
        depth_weight    : float       = 1,
        edge_weight     : float       = 1,
        use_pseudo_gt   : bool        = False,
        number_refs     : int         = 2,
        *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.num_scales  = len(window_size)
        self.use_pgt     = use_pseudo_gt
        self.number_refs = number_refs
        weight_decay     = [0.1, 0.0001, 0.001]
        
        if color_space == "rgb":
            mlp = MLP_RGB
        elif color_space == "rgb_d":
            mlp = MLP_RGB_D
        elif color_space == "hsv_v":
            mlp = MLP_HSV_V
        else:
            raise ValueError(f"Invalid color space: {color_space}")
        
        self.mlp = mlp(
            window_size      = window_size,
            hidden_channels  = hidden_channels,
            down_size        = down_size,
            hidden_layers    = hidden_layers,
            out_layers       = out_layers,
            omega_0          = omega_0,
            first_bias_scale = first_bias_scale,
            nonlinear        = nonlinear,
            weight_decay     = weight_decay,
            dba_eps          = dba_eps,
            gf_radius        = gf_radius,
            denoise          = denoise,
            denoise_ksize    = denoise_ksize,
            denoise_color    = denoise_color,
            denoise_space    = denoise_space,
        )
        self.pseudo_gt_generator = utils.PseudoGTGenerator(
            number_refs   = self.number_refs,
            gamma_upper   = -2,
            gamma_lower   =  3,
            exposed_level =  0.5,
            pool_size     =  25,
        )
        self.saved_input     = None
        self.saved_pseudo_gt = None
        
        # Loss
        if loss_hsv and "hsv" in color_space:
            self.loss = LossHSV(
                exp_mean     = 1.0 - exp_mean,
                # exp_weight   = exp_weight,
                # spa_weight   = spa_weight,
                # tv_weight    = tv_weight,
                depth_weight = depth_weight,
                edge_weight  = edge_weight,
                color_weight = color_weight,
            )
        else:
            self.loss = Loss(
                exp_mean     = exp_mean,
                exp_weight   = exp_weight,
                spa_weight   = spa_weight,
                color_weight = color_weight,
                tv_weight    = tv_weight,
                depth_weight = depth_weight,
                edge_weight  = edge_weight,
            )
        self.loss_recon = nn.MSELoss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    def compute_efficiency_score(
        self,
        image_size: _size_2_t = 512,
        channels  : int       = 3,
        runs      : int       = 1000,
        verbose   : bool      = False,
    ) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        # Define input tensor
        h, w      = core.get_image_size(image_size)
        datapoint = {
            "image": torch.rand(1, channels, h, w).to(self.device),
            "depth": torch.rand(1,        1, h, w).to(self.device)
        }
        
        # Get FLOPs and Params
        flops, params = core.custom_profile(self, inputs=datapoint, verbose=verbose)
        # flops         = FlopCountAnalysis(self, datapoint).total() if flops == 0 else flops
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params
        
        # Get time
        timer = core.Timer()
        for i in range(runs):
            timer.tick()
            _ = self.forward(datapoint)
            timer.tock()
        avg_time = timer.avg_time
        
        # Print
        if verbose:
            console.log(f"FLOPs (G) : {flops:.4f}")
            console.log(f"Params (M): {params:.4f}")
            console.log(f"Time (s)  : {avg_time:.17f}")
        
        return flops, params, avg_time
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        if self.use_pgt:
            # Saving n-th input and n-th pseudo gt
            nth_input     = datapoint
            nth_output    = self.forward(datapoint=datapoint, *args, **kwargs)
            nth_image     = nth_output["image"]
            nth_enhanced  = nth_output["enhanced"].clone().detach()
            nth_pseudo_gt = self.pseudo_gt_generator(nth_image, nth_enhanced)
            if self.saved_input is not None:
                # Getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handled automatically by pytorch lightning)
                outputs  = self.forward(datapoint=datapoint, *args, **kwargs)
                image    = outputs["image"]
                enhanced = outputs["enhanced"]
                loss_enh = 0
                for i in range(self.num_scales):
                    image_lr_i     = outputs[f"image_lr_{i}"]
                    illu_lr_i      = outputs[f"illu_lr_{i}"]
                    enhanced_lr_i  = outputs[f"enhanced_lr_{i}"]
                    depth_lr_i     = outputs[f"depth_lr_{i}"]
                    loss_i         = self.loss(image, image_lr_i, illu_lr_i, enhanced, enhanced_lr_i, depth_lr_i)
                    loss_enh      += loss_i
                pseudo_gt  = self.saved_pseudo_gt
                loss_recon = self.loss_recon(enhanced, pseudo_gt)
                loss       = loss_recon + loss_enh  # * 5
                outputs["loss"] = loss
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs  = self.forward(datapoint=datapoint, *args, **kwargs)
            image    = outputs["image"]
            enhanced = outputs["enhanced"]
            loss     = 0
            for i in range(self.num_scales):
                image_lr_i     = outputs[f"image_lr_{i}"]
                illu_lr_i      = outputs[f"illu_lr_{i}"]
                enhanced_lr_i  = outputs[f"enhanced_lr_{i}"]
                depth_lr_i     = outputs[f"depth_lr_{i}"]
                loss_i         = self.loss(image, image_lr_i, illu_lr_i, enhanced, enhanced_lr_i, depth_lr_i)
                loss          += loss_i
            outputs["loss"] = loss
        return outputs
        
    def forward(self, datapoint: dict, n_iters: int = 1, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        depth = datapoint.get("depth")
        for i in range(n_iters):
            outputs = self.mlp(image, depth)
            image   = outputs["enhanced"]
        # Return
        return outputs
       
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 300,   # 500
        lr           : float = 1e-5,  # 1e-5
        weight_decay : float = 3e-4,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay
            )
        
        # Pre-processing
        self.saved_input     = None
        self.saved_pseudo_gt = None
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint, n_iters=1)
            optimizer.zero_grad()
            loss    = outputs["loss"]
            if loss is not None:
                loss.backward(retain_graph=True)
                optimizer.step()
            # if self.verbose:
            #    console.log(f"Loss: {loss.item()}")
            
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint, n_iters=1)
        timer.tock()
        self.assert_outputs(outputs)
        
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
# endregion
