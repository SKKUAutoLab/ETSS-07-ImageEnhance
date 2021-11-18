#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Asymmetric loss.

Paper: https://arxiv.org/abs/2009.14119.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from .builder import LOSSES

logger = logging.getLogger()


@LOSSES.register(name="asymmetric_loss_multilabel")
class AsymmetricLossMultiLabel(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        gamma_neg: int   = 4,
        gamma_pos: int   = 1,
        clip	 : float = 0.05,
        eps 	 : float = 1e-8,
        disable_torch_grad_focal_loss: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip 	   = clip
        self.eps 	   = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
    # MARK: Forward Pass
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            y_hat (torch.Tensor):
                Input logits.
            y (torch.Tensor):
                Targets (multi-label binarized vector).
        """
        # Calculating Probabilities
        y_hat_sigmoid = torch.sigmoid(y_hat)
        y_hats_pos    = y_hat_sigmoid
        y_hats_neg	  = 1 - y_hat_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (y_hats_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(y_hats_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 		 	= y_hats_pos * y
            pt1 			= xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt  		    = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w 	= torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


@LOSSES.register(name="asymmetric_loss_single_label")
class AsymmetricLossSingleLabel(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        gamma_pos: int	 = 1,
        gamma_neg: int 	 = 4,
        eps		 : float = 0.1,
        reduction: str 	 = "mean"
    ):
        super().__init__()
        self.eps 			 = eps
        self.logsoftmax      = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos 		 = gamma_pos
        self.gamma_neg 		 = gamma_neg
        self.reduction 		 = reduction

    # MARK: Forward Pass

    def forward(
		self, y_hat: torch.Tensor, y: torch.Tensor, reduction=None
	) -> torch.Tensor:
        """Forward pass.
        
        Args:
            y_hat (torch.Tensor):
                Input logits.
            y (torch.Tensor):
                Targets (1-hot vector).
        """
        num_classes 		 = y_hat.size()[-1]
        log_preds   		 = self.logsoftmax(y_hat)
        targets		         = torch.zeros_like(y_hat)
        self.targets_classes = targets.scatter_(1, y.long().unsqueeze(1), 1)

        # ASL weights
        targets		 = self.targets_classes
        anti_targets = 1 - targets
        xs_pos 		 = torch.exp(log_preds)
        xs_neg 		 = 1 - xs_pos
        xs_pos 	     = xs_pos * targets
        xs_neg		 = xs_neg * anti_targets
        asymmetric_w = torch.pow(
			1 - xs_pos - xs_neg,
			self.gamma_pos * targets + self.gamma_neg *  anti_targets
		)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # Loss calculation
        loss = - self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss
