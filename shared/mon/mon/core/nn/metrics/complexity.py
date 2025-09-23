#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements efficiency score metrics."""

__all__ = [
    "benchmark",
    "compute_complexity",
]

import thop
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

from mon.core.console import log
from mon.core.device import get_model_device
from mon.core.dtypes import image as I


def compute_complexity(model: nn.Module, imgsz: int = 512, channels: int = 3) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.
        channels: Number of input channels. Default: ``3``.

    Returns:
        A tuple of :math:`(flops, params)`.
    """
    h, w   = I.imgsz(imgsz)
    input  = torch.rand(1, channels, h, w).to(get_model_device(model))
    flops, params = thop.profile(model, inputs=(input,), verbose=False)
    flops  = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params = model.params           if hasattr(model, "params") and params == 0 else params
    params = parameter_count(model) if hasattr(model, "params") else params
    params = sum(params.values())   if isinstance(params, dict) else params
    return flops, params


def benchmark(model: nn.Module, imgsz: int = 512, channels: int = 3):
    """Measures and logs the complexity of a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.
        channels: Number of input channels. Default: ``3``.
    """
    flops, params = compute_complexity(model=model, imgsz=imgsz, channels=channels)
    log(f"Params    : {params:.4f}")
    log(f"FLOPs     : {flops:.4f}")
