#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Restoration of Underexposed Images via Robust
Retinex Decomposition," ICME 2020.

References:
    - https://github.com/aaaaangel/RRDNet
"""

__all__ = [
    "RRDNet",
]

from typing import Literal

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]

bilateral_ksize = (3, 3)
bilateral_color = 0.1
bilateral_space = (1.5, 1.5)


# ----- Loss -----
class Loss(nn.Loss):
    
    def __init__(
        self,
        illu_factor   : float = 1,
        reflect_factor: float = 1,
        noise_factor  : float = 5000,
        reduction     : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction)
        self.illu_factor    = illu_factor
        self.reflect_factor = reflect_factor
        self.noise_factor   = noise_factor
        
    def forward(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        loss_recons   =      self.reconstruction_loss(image, illumination, reflectance, noise)
        loss_illu     = self.illumination_smooth_loss(image, illumination, reflectance, noise)
        loss_reflect  =  self.reflectance_smooth_loss(image, illumination, reflectance, noise)
        loss_noise    =               self.noise_loss(image, illumination, reflectance, noise)
        loss          = (
              loss_recons
            + loss_illu     * self.illu_factor
            + loss_reflect  * self.reflect_factor
            + loss_noise    * self.noise_factor
        )
        return loss
    
    def reconstruction_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        reconstructed_image = illumination * reflectance + noise
        return torch.norm(image - reconstructed_image, 1)
    
    def illumination_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        g_kernel_size   = 5
        g_padding       = 2
        sigma           = 3
        kx              = cv2.getGaussianKernel(g_kernel_size, sigma)
        ky              = cv2.getGaussianKernel(g_kernel_size, sigma)
        gaussian_kernel = np.multiply(kx, np.transpose(ky))
        gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(image.device)
        gray_tensor     = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        max_rgb, _      = torch.max(image, 1)
        max_rgb         = max_rgb.unsqueeze(1)
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_illu_h, gradient_illu_w = self.gradient(illumination)
        weight_h = 1 / (F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding) + 0.0001)
        weight_w = 1 / (F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding) + 0.0001)
        weight_h.detach()
        weight_w.detach()
        loss_h = weight_h * gradient_illu_h
        loss_w = weight_w * gradient_illu_w
        max_rgb.detach()
        return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)
    
    def reflectance_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        reffac      = 1
        gray_tensor = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        gradient_gray_h,    gradient_gray_w    = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_reflect_h, gradient_reflect_w = self.gradient(reflectance)
        weight = 1 / (illumination * gradient_gray_h * gradient_gray_w + 0.0001)
        weight = self.normalize01(weight)
        weight.detach()
        loss_h           = weight * gradient_reflect_h
        loss_w           = weight * gradient_reflect_w
        refrence_reflect = image / illumination
        refrence_reflect.detach()
        return loss_h.sum() + loss_w.sum() + reffac*torch.norm(refrence_reflect - reflectance, 1)
    
    def noise_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        weight_illu = illumination
        weight_illu.detach()
        loss = weight_illu * noise
        return torch.norm(loss, 2)
    
    def gradient(self, img):
        height      = img.size(2)
        width       = img.size(3)
        gradient_h  = (img[:, :, 2:, :] - img[:, :, :height - 2, :]).abs()
        gradient_w  = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
        gradient_h  = F.pad(gradient_h, [0, 0, 1, 1], "replicate")
        gradient_w  = F.pad(gradient_w, [1, 1, 0, 0], "replicate")
        gradient2_h = (img[:, :, 4:, :] - img[:, :, :height - 4, :]).abs()
        gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], "replicate")
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], "replicate")
        return gradient_h * gradient2_h, gradient_w * gradient2_w
    
    def normalize01(self, img):
        minv = img.min()
        maxv = img.max()
        return (img-minv)/(maxv-minv)
    
    def gaussianblur3(self, input):
        g_kernel_size   = 5
        g_padding       = 2
        sigma           = 3
        kx              = cv2.getGaussianKernel(g_kernel_size, sigma)
        ky              = cv2.getGaussianKernel(g_kernel_size, sigma)
        gaussian_kernel = np.multiply(kx, np.transpose(ky))
        gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(input.device)
        slice1 = F.conv2d(input[:, 0, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        slice2 = F.conv2d(input[:, 1, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        slice3 = F.conv2d(input[:, 2, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        x      = torch.cat([slice1, slice2, slice3], dim=1)
        return x


# ----- Model -----
@MODELS.register(name="rrdnet", arch="rrdnet")
class RRDNet(base.ImageEnhancementModel):
    """RRDNet model for low-light image enhancement."""
    
    arch     : str          = "rrdnet"
    name     : str          = "rrdnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        gamma         : float = 0.4,
        illu_factor   : float = 1,
        reflect_factor: float = 1,
        noise_factor  : float = 5000,
        iters         : int   = 1000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gamma          = gamma
        self.illu_factor    = illu_factor
        self.reflect_factor = reflect_factor
        self.noise_factor   = noise_factor
        self.iters          = iters
        
        # Network
        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.noise_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        
        # Optimizer
        self.configure_optimizers()
        
        # Loss
        self.loss = Loss(illu_factor=illu_factor, reflect_factor=reflect_factor, noise_factor=noise_factor)
        
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
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        image        = outputs["image"]
        illumination = outputs["illumination"]
        reflectance  = outputs["reflectance"]
        noise        = outputs["noise"]
        loss         = self.loss(image, illumination, reflectance, noise)
        
        return outputs | {
            "loss": loss,
        }
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Input
        image = datapoint["image"]
        
        # Enhance
        illumination = torch.sigmoid(self.illumination_net(image))
        reflectance  = torch.sigmoid(self.reflectance_net(image))
        noise        = torch.tanh(self.noise_net(image))
        adjust_illu  = torch.pow(illumination, self.gamma)
        enhanced     = adjust_illu * ((image - noise) / illumination)
        enhanced     = torch.clamp(enhanced, min=0, max=1)

        return {
            "image"       : image,
            "illumination": illumination,
            "reflectance" : reflectance,
            "noise"       : noise,
            "enhanced"    : enhanced
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
        optimizer = optimizer or nn.Adam(self.parameters(), lr=0.001)
            
        # Input
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Optimize
        timer = core.Timer()
        timer.tick()
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint=datapoint)
            self.zero_grad()
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
        self.eval()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        
        # Return
        return outputs | {
            "time": timer.avg_time,
        }
