#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements color, exposure, and illumination consistency.

These losses focus on maintaining or correcting color properties, illumination, or
exposure, ensuring natural appearance in enhanced or restored images.
"""

__all__ = [
    "ColorConstancyLoss",
    "DepthAwareIlluminationLoss",
    "EdgeAwareIlluminationLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "TotalVariationLoss",
]

from typing import Literal

import torch
from torch.nn.common_types import _size_2_t

from mon.constants import LOSSES
from mon.nn.loss import base


# ----- Color Loss -----
@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(base.Loss):
    """Color Constancy Loss corrects potential color deviations in the enhanced image
    and builds relations among the three adjusted channels.

    Args:
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``.
        eps: Small constant to avoid sqrt by zero as ``float``. Default is ``1e-6``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9
    """
    
    def __init__(self, eps: float = 1e-6, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        self.eps = eps
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the color constancy loss for the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, 3, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(torch.abs(mr - mg), 2)
        d_rb       = torch.pow(torch.abs(mr - mb), 2)
        d_gb       = torch.pow(torch.abs(mb - mg), 2)
        d_rg2      = torch.pow(d_rg, 2)
        d_rb2      = torch.pow(d_rb, 2)
        d_gb2      = torch.pow(d_gb, 2)
        loss       = d_rg2 + d_rb2 + d_gb2
        loss       = torch.pow(loss + self.eps, 0.5)
        loss       = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    

# ----- Exposure Loss -----
@LOSSES.register(name="exposure_control_loss")
class ExposureControlLoss(base.Loss):
    """Exposure Control Loss measures the distance between the average intensity value
    of a local region and the well-exposedness level E.

    Args:
        patch_size: Kernel size for pooling layer as ``int`` or ``tuple[int, int]``.
            Default is ``16``.
        mean_val: Well-exposedness level E as ``float``. Default is ``0.6``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """
    
    def __init__(
        self,
        patch_size: _size_2_t = 16,
        mean_val  : float     = 0.6,
        reduction : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = torch.nn.AvgPool2d(self.patch_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = torch.mean(input, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="exposure_value_control_loss")
class ExposureValueControlLoss(base.Loss):
    """Exposure Value Control Loss measures the absolute value of the ``ExposureControlLoss``.

    Args:
        patch_size: Kernel size for pooling layer as ``int`` or ``tuple[int, int]``.
            Default is ``16``.
        mean_val: Well-exposedness level E as ``float``; lower values produce
            brighter images. Default is ``0.6``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
        Default is ``"mean"``.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """
    
    def __init__(
        self,
        patch_size: _size_2_t = 16,
        mean_val  : float     = 0.6,
        reduction : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = torch.nn.AvgPool2d(self.patch_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes the exposure value control loss for the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``
        """
        x    = torch.mean(input, 1, keepdim=True)  # Channel-wise mean: [B, 1, H, W]
        mean = self.pool(x) ** 0.5                 # Pooled mean: [B, 1, H', W']
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = torch.abs(torch.mean(loss))
        # loss = base.reduce_loss(loss=diff, reduction=self.reduction)  # Reduced absolute difference
        return loss


# ----- Illumination Loss -----
@LOSSES.register(name="depth_aware_illumination_loss")
class DepthAwareIlluminationLoss(base.Loss):
    """Calculate the depth-weighted smoothness loss for 4D tensors.

    Args:
        alpha: Weighting factor for depth influence as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        alpha    : float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(reduction=reduction)
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Computes the depth-weighted illumination smoothness loss.

        Args:
            input: Illumination tensor as ``torch.Tensor`` with shape [B, C, H, W].
            depth: Depth tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        # Calculate gradients of illumination map (L) in x and y directions
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        
        # Calculate gradients of depth map (D) in x and y directions
        D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        # Compute depth-weighted terms for x and y directions
        weight_dx = torch.exp(-self.alpha * torch.abs(D_dx))
        weight_dy = torch.exp(-self.alpha * torch.abs(D_dy))
        
        # Apply depth weights to illumination gradients and take the mean
        loss_dx = torch.mean(weight_dx * torch.abs(L_dx))
        loss_dy = torch.mean(weight_dy * torch.abs(L_dy))
        
        # Sum the losses from both directions
        return loss_dx + loss_dy


@LOSSES.register(name="edge_aware_illumination_loss")
class EdgeAwareIlluminationLoss(base.Loss):
    """Edge-Aware Illumination Loss penalizes illumination changes along strong edges.

    Args:
        beta: Weighting factor for edge influence as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        beta     : float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(reduction=reduction)
        self.beta = beta
    
    def forward(self, input: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Computes the edge-aware illumination smoothness loss.

        Args:
            input: Illumination tensor as ``torch.Tensor`` with shape [B, C, H, W].
            edge: Edge tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        # Calculate gradients of illumination map (L) in x and y directions
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        
        # Calculate gradients of edge map (E) in x and y directions
        E_dx = edge[:, :, :, 1:] - edge[:, :, :, :-1]
        E_dy = edge[:, :, 1:, :] - edge[:, :, :-1, :]
        
        # Apply edge weights to illumination gradients; areas with stronger edges have lower weight
        # weight_dx = torch.exp(-torch.abs(E_dx))
        # weight_dy = torch.exp(-torch.abs(E_dy))
        weight_dx = 1 - self.beta * torch.abs(E_dx)
        weight_dy = 1 - self.beta * torch.abs(E_dy)
        
        # Calculate edge-aware losses by penalizing illumination changes along strong edges
        loss_dx = torch.mean(weight_dx * torch.abs(L_dx))
        loss_dy = torch.mean(weight_dy * torch.abs(L_dy))
        
        # Sum the losses from both directions
        return loss_dx + loss_dy


@LOSSES.register(name="total_variation_loss")
class TotalVariationLoss(base.Loss):
    """Total Variation Loss on the Illumination (Illumination Smoothness Loss) preserves
    monotonicity relations between neighboring pixels to avoid aggressive and sharp changes.

    Args:
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """Computes the total variation loss for the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        x = input
        b, _, h_x, w_x = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])  # (x.size()[2]-1) * x.size()[3]
        count_w = self._tensor_size(x[:, :, :, 1:])  # x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[:, :, 1:,  :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss    = 2 * (h_tv / count_h + w_tv / count_w) / b
        return loss
        
    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        """Computes the total number of elements in the tensor.

        Args:
            t: Input tensor as ``torch.Tensor``.

        Returns:
            Number of elements as ``int``.
        """
        return t.size()[1] * t.size()[2] * t.size()[3]
