#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image feature priors (structural and semantic priors).

This category involves identifying and isolating structural or semantic features
(e.g., edges, boundaries) in an image.
"""

__all__ = [
    "BoundaryAwarePrior",
    "boundary_aware_prior",
]

import kornia
import torch
import torch.nn as nn


def boundary_aware_prior(
    image      : torch.Tensor,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor:
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        image: An RGB or grayscale image as a ``torch.Tensor`` of
            shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, returns the gradient image instead of binary boundary.
            Default: ``False``.
        normalized: L1 norm of the kernel is set to 1 if ``True``. Default: ``False``.

    Returns:
        The boundary prior with similar type and format as the input ``image``.

    Raises:
        ValueError: If ``image`` type is not supported.
    """
    image    = image.to(torch.float32)
    gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
    g_max    = torch.max(gradient)
    gradient = gradient / g_max
    boundary = (gradient > eps).float()
    
    # return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, returns the gradient image instead of binary boundary.
            Default: ``False``.
        normalized: L1 norm of the kernel is set to 1 if ``True``. Default: ``False``.
    """
    
    def __init__(self, eps: float = 0.05, as_gradient: bool = False, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.as_gradient = as_gradient
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return boundary_aware_prior(image, self.eps, self.as_gradient, self.normalized)
