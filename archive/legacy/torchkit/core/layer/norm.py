#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization Layers.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.core.layer.builder import NORM_LAYERS

logger = logging.getLogger()


# MARK: - Register

NORM_LAYERS.register(name="batch_norm",       module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm_1d",    module=nn.BatchNorm1d)
NORM_LAYERS.register(name="batch_norm_2d",    module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm_3d",    module=nn.BatchNorm3d)
# NORM_LAYERS.register(name="SyncBN", module=SyncBatchNorm)
# NORM_LAYERS.register(name="group_norm",       module=nn.GroupNorm)
NORM_LAYERS.register(name="layer_norm",       module=nn.LayerNorm)
NORM_LAYERS.register(name="instance_norm",    module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm_1d", module=nn.InstanceNorm1d)
NORM_LAYERS.register(name="instance_norm_2d", module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm_3d", module=nn.InstanceNorm3d)


# MARK: - GroupNorm

@NORM_LAYERS.register(name="group_norm")
class GroupNorm(nn.GroupNorm):
    """NOTE `num_channels` is swapped to first arg for consistency in swapping
    norm layers with BN.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_channels: int,
        num_groups  : int,
        eps         : float = 1e-5,
        affine      : bool  = True
    ):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, self.num_groups, self.weight, self.bias,
                            self.eps)


# MARK: - LayerNorm2d

@NORM_LAYERS.register(name="layer_norm_2d")
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of `2D` spatial BCHW tensors."""

    # MARK: Magic Functions

    def __init__(self, num_channels: int):
        super().__init__(num_channels)

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
			self.bias, self.eps
        ).permute(0, 3, 1, 2)
