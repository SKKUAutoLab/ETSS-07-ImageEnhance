#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger()


# MARK: - SpaceToDepth

class SpaceToDepth(nn.Module):

    # MARK: Magic Functions

    def __init__(self, block_size: int = 4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()                    # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)   # (N, C*bs^2, H//bs, W//bs)
        return x


# MARK: - SpaceToDepthJit

@torch.jit.script
class SpaceToDepthJit(object):

    # MARK: Magic Functions

    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)        # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)         # (N, C*bs^2, H//bs, W//bs)
        return x


# MARK: - SpaceToDepthModule

class SpaceToDepthModule(nn.Module):

    # MARK: Magic Functions

    def __init__(self, no_jit: bool = False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# MARK: - DepthToSpace

class DepthToSpace(nn.Module):

    # MARK: Magic Functions

    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    # MARK: Forward Pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)    # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()                  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
