#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Fast Context-Based Low-Light Image Enhancement via Neural
Implicit Representations," ECCV 2024.

References:
    - https://github.com/ctom2/colie
"""

__all__ = [
    "CoLIE_RE",
]

from typing import Literal

import kornia
import numpy as np
import torch
from torch.nn import functional as F

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t
from mon.vision import filtering, types
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Loss -----
class Loss(nn.Loss):
    
    def __init__(
        self,
        L        : float = 0.5,
        alpha    : float = 1,
        beta     : float = 20,
        gamma    : float = 8,
        delta    : float = 5,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.l_exp = nn.ExposureValueControlLoss(patch_size=16, mean_val=L)
        self.l_tv  = nn.TotalVariationLoss()
        
    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_v_lr      : torch.Tensor,
        image_v_fixed_lr: torch.Tensor,
    ) -> torch.Tensor:
        loss_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
        loss_tv       = self.l_tv(illu_lr)
        loss_exp      = torch.mean(self.l_exp(illu_lr))
        loss_sparsity = torch.mean(image_v_fixed_lr)
        loss = (
                   loss_spa * self.alpha
                  + loss_tv * self.beta
                 + loss_exp * self.gamma
            + loss_sparsity * self.delta
        )
        return loss


# ----- Model -----
@MODELS.register(name="colie_re", arch="colie")
class CoLIE_RE(base.ImageEnhancementModel):
    """COLIE model for low-light image enhancement.
    
    Args:
        window_size: Context window size. Default is ``1``.
        down_size  : Downsampling size. Default is ``256``.
        add_layer: Should be in range of [1, `num_layers` - 2].
    
    References:
        - https://github.com/ctom2/colie
    """
    
    arch     : str          = "colie"
    name     : str          = "colie_re"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        window_size : int         = 7,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        iters       : int         = 100,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.iters       = iters
        
        # Network
        patch_layers   = [nn.INRLayer(self.patch_dim, hidden_dim, "sine", w0=30.0, is_first=True)]
        spatial_layers = [nn.INRLayer(2, hidden_dim, "sine", w0=30.0, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim, "sine", w0=30.0))
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, "sine", w0=30.0))
        patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim // 2, "sine", w0=30.0))
        spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim // 2, "sine", w0=30.0))
        
        output_layers  = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, "sine", w0=30.0))
        output_layers.append(nn.INRLayer(hidden_dim, 1, "sine", w0=30.0, is_last=True))
        
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
        # Optimizer
        self.configure_optimizers()
        
        # Loss
        
        self.loss = Loss(
            L     = self.loss["L"],
            alpha = self.loss["alpha"],
            beta  = self.loss["beta"],
            gamma = self.loss["gamma"],
            delta = self.loss["delta"]
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes weights for the model.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    def compute_efficiency_score(self, image_size: _size_2_t = 512) -> tuple[float, float]:
        """Compute model efficiency score (FLOPs, params).

        Args:
            image_size: Input size as ``int`` or [H, W]. Default is ``512``.
            channels: Number of input channels as ``int``. Default is ``3``.

        Returns:
            Tuple of (FLOPs, parameter count) as ``float`` values.
        """
        from fvcore.nn import parameter_count
        
        h, w      = types.image_size(image_size)
        datapoint = {"image": torch.rand(1, 3, h, w).to(self.device)}
        
        flops, params = core.thop.custom_profile(self, inputs=datapoint, verbose=False)
        # flops         = FlopCountAnalysis(self, datapoint).total() if flops == 0 else flops
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
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        illu_lr          = outputs["illu_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        loss             = self.loss(illu_lr, image_v_lr, image_v_fixed_lr)
        
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
            ["image_hsv", "image_v", "image_v_lr", "patch", "spatial"],
            datapoint
        ):
            datapoint = self.prepare_input(datapoint)
        image_v_lr  = datapoint["image_v_lr"]
        patch       = datapoint["patch"]
        spatial     = datapoint["spatial"]
        
        # Mapping
        illu_res_lr = self.output_net(torch.cat([self.patch_net(patch), self.spatial_net(spatial)], -1))
        illu_res_lr = illu_res_lr.view(1, 1, self.down_size, self.down_size)
        
        # Enhance
        illu_lr          = illu_res_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_lr + 1e-4)
        
        return {
            "illu_lr"         : illu_lr,
            "image_v_lr"      : image_v_lr,
            "image_v_fixed_lr": image_v_fixed_lr,
        }
        
    def prepare_input(self, datapoint: dict) -> dict:
        """Prepares input for the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of prepared inputs.
        """
        image_rgb  = datapoint["image"]
        image_hsv  = kornia.color.rgb_to_hsv(image_rgb)
        image_v    = image_hsv.clone()[:, 2:3, :, :]
        image_v_lr = self.interpolate_image(image_v)
        patch      = self.create_patches(image_v_lr)
        spatial    = self.create_coords()
        return datapoint | {
            "image_hsv" : image_hsv,
            "image_v"   : image_v,
            "image_v_lr": image_v_lr,
            "patch"     : patch,
            "spatial"   : spatial,
        }
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def create_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = types.image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(self.device)
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
    
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)
    
    def create_coords(self) -> torch.Tensor:
        """Creates a coordinates grid."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float().to(self.device)
        return coords
    
    @staticmethod
    def filter_up(x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor, radius: int = 1):
        """Applies the guided filter to upscale the predicted image. """
        gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0, 1)
        return y_hr
    
    @staticmethod
    def replace_v_component(image_hsv: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `V` component of an HSV image [1, 3, H, W]."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
    @staticmethod
    def replace_i_component(image_hvi: torch.Tensor, i_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `I` component of an HVI image [1, 3, H, W]."""
        image_hvi[:, 2, :, :] = i_new
        return image_hvi
    
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
        optimizer = optimizer or nn.Adam(self, lr=0.00001, weight_decay=0.0003)
        
        # Input
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        datapoint = self.prepare_input(datapoint)
        
        # Optimize
        timer = core.Timer()
        timer.tick()
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
        self.eval()
        outputs = self.forward(datapoint=datapoint)
        image_hsv        = datapoint["image_hsv"].clone()
        image_v          = datapoint["image_v"]
        image_v_lr       = outputs["image_v_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = kornia.color.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        timer.tock()
        
        # Return
        return outputs | {
            "enhanced": image_rgb_fixed,
            "time"    : timer.avg_time,
        }
