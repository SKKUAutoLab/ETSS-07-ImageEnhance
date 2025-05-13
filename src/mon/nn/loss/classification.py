#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions for classification tasks."""

__all__ = [
    "DiceLoss",
]

from typing import Literal

import torch

from mon.constants import LOSSES
from mon.nn.loss import base


# region Dice Loss

def dice_coefficient(
    input       : torch.Tensor,
    target      : torch.Tensor,
    reduce_batch: bool  = False,
    eps        : float = 1e-6
) -> torch.Tensor:
    """Computes Dice coefficient for input and target tensors.

    Args:
        input: Prediction tensor as ``torch.Tensor``.
        target: Ground truth tensor as ``torch.Tensor``.
        reduce_batch: Reduces batch dim if ``True``. Default is ``False``.
        eps: Smoothing factor. Default is ``1e-6``.

    Returns:
        Dice coefficient as ``torch.Tensor``.

    Raises:
        ValueError: If sizes mismatch or dims invalid with ``reduce_batch``.
    """
    if input.size() != target.size():
        raise ValueError(f"[input] and [target] must have same size, got "
                         f"{input.size()} and {target.size()}.")
    if input.dim() != 3 and reduce_batch:
        raise ValueError(f"[input] must have 3 dims, got {input.dim()}.")
    
    sum_dim  = (-1, -2) if input.dim() == 2 or not reduce_batch else (-1, -2, -3)
    inter    = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice     = (inter + eps) / (sets_sum + eps)
    return dice


def multiclass_dice_coefficient(
    input       : torch.Tensor,
    target      : torch.Tensor,
    reduce_batch: bool  = False,
    eps         : float = 1e-6
) -> torch.Tensor:
    """Computes Dice coefficient for multiclass tensors.

    Args:
        input: Prediction tensor as ``torch.Tensor``.
        target: Ground truth tensor as ``torch.Tensor``.
        reduce_batch: Reduces batch dim if ``True``. Default is ``False``.
        eps: Smoothing factor. Default is ``1e-6``.

    Returns:
        Average Dice coefficient as ``torch.Tensor``.
    """
    return dice_coefficient(
        input        = input.flatten(0, 1),
        target       = target.flatten(0, 1),
        reduce_batch = reduce_batch,
        eps= eps,
    )


@LOSSES.register(name="dice_loss")
class DiceLoss(base.Loss):
    """Dice loss for binary or multiclass classification tasks.

    Args:
        loss_weight: Weight applied to the loss. Default is ``1.0``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default is ``"mean"``.
        reduce_batch: Reduces batch dimension if ``True``. Default is ``True``.
        multiclass: Uses multiclass Dice if ``True``. Default is ``False``.
    """
    
    def __init__(
        self,
        reduction   : Literal["none", "mean", "sum"] = "mean",
        reduce_batch: bool  = True,
        multiclass  : bool  = False
    ):
        super().__init__(reduction=reduction)
        self.reduce_batch = reduce_batch
        self.multiclass   = multiclass
        self.fn = (multiclass_dice_coefficient if multiclass else dice_coefficient)

    def forward(self, input : torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss.

        Args:
            input: Predicted tensor as ``torch.Tensor``.
            target: Target tensor as ``torch.Tensor``.

        Returns:
            Reduced loss as ``torch.Tensor``.
        """
        loss = 1 - self.fn(input=input, target=target, reduce_batch=self.reduce_batch)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss

# endregion
