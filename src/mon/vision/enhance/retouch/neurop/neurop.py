#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeurOP.

This module implements the paper: "Neural Color Operators for Sequential Image
Retouching".

References:
    https://github.com/amberwangyili/neurop
"""

from __future__ import annotations

__all__ = [
    "NeurOPInit",
    "NeurOP_RE",
]

from typing import Any, Literal

import lightning.pytorch.utilities.types
import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.nn import functional as F
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
StepOutput   = lightning.pytorch.utilities.types.STEP_OUTPUT


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        pixel_weight: float = 10.0,
        reduction   : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.loss_pix   = nn.L1Loss(reduction="mean")
        self.loss_cos   = nn.CosineSimilarityLoss(reduction="mean")
        self.loss_tv    = nn.TotalVariationLoss(reduction="mean")
        self.loss_ratio = 1 / pixel_weight
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_pix = self.loss_pix(input, target)
        loss_cos = self.loss_cos(input, target)
        loss_tv  = self.loss_tv(input)
        loss     = loss_pix + self.loss_ratio * (loss_cos + loss_tv)
        return loss
    
# endregion


# region Modules

class Operator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_nf: int = 64):
        super().__init__()
        self.base_nf      = base_nf
        self.out_channels = out_channels
        self.encoder      = nn.Conv2d(in_channels, base_nf,      1, 1)
        self.mid_conv     = nn.Conv2d(base_nf,     base_nf,      1, 1)
        self.decoder      = nn.Conv2d(base_nf,     out_channels, 1, 1)
        self.act          = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        x_code = self.encoder(x)
        y_code = x_code + val
        y_code = self.act(self.mid_conv(y_code))
        y      = self.decoder(y_code)
        return y


class Renderer(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_nf: int = 64):
        super().__init__()
        self.in_channels  = in_channels
        self.base_nf      = base_nf
        self.out_channels = out_channels
        self.ex_block     = Operator(in_channels, out_channels, base_nf)
        self.bc_block     = Operator(in_channels, out_channels, base_nf)
        self.vb_block     = Operator(in_channels, out_channels, base_nf)
        
    def forward(
        self,
        x_ex: torch.Tensor,
        x_bc: torch.Tensor,
        x_vb: torch.Tensor,
        v_ex: torch.Tensor,
        v_bc: torch.Tensor,
        v_vb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec_ex = self.ex_block(x_ex, 0)
        rec_bc = self.bc_block(x_bc, 0)
        rec_vb = self.vb_block(x_vb, 0)
        
        map_ex = self.ex_block(x_ex, v_ex)
        map_bc = self.bc_block(x_bc, v_bc)
        map_vb = self.vb_block(x_vb, v_vb)
        
        return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb


class Encoder(nn.Module):
    
    def __init__(self, in_channels=3, encode_nf=32):
        super().__init__()
        stride     = 2
        pad        = 0
        self.pad   = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf,   encode_nf, 3, stride, pad, bias=True)
        self.act   = nn.ReLU(inplace=True)
        self.max   = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.size()
        conv1_out  = self.act(self.conv1(self.pad(x)))
        conv2_out  = self.act(self.conv2(self.pad(conv1_out)))
        std, mean  = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
        maxs       = self.max(conv2_out).squeeze(2).squeeze(2)
        out        = torch.cat([std, mean, maxs], dim=1)
        return out


class Predictor(nn.Module):
    
    def __init__(self, fea_dim: int):
        super().__init__()
        self.fc3  = nn.Linear(fea_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, img_fea: torch.Tensor) -> torch.Tensor:
        val = self.tanh(self.fc3(img_fea))
        return val

# endregion


# region Models

@MODELS.register(name="neurop_init", arch="neurop")
class NeurOPInit(base.ImageEnhancementModel):
    """Neural Color Operators for Sequential Image Retouching.
    
    References:
    https://github.com/amberwangyili/neurop
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "neurop"
    tasks    : list[Task]   = [Task.LLIE, Task.RETOUCH]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        out_channels: int = 3,
        base_nf     : int = 64,
        encode_nf   : int = 32,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "neurop_init",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        self.renderer = Renderer(in_channels, out_channels, base_nf)
        
        # Loss functions
        self.loss = nn.L1Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        pass
        
    def assert_outputs(self, outputs: dict) -> bool:
        pass
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        # Loss
        image_ex        = datapoint["image_ex"]
        image_bc        = datapoint["image_bc"]
        image_vb        = datapoint["image_vb"]
        ref_ex          = datapoint["ref_ex"]
        ref_bc          = datapoint["ref_bc"]
        ref_vb          = datapoint["ref_vb"]
        rec_image_ex    = outputs["rec_image_ex"]
        rec_image_bc    = outputs["rec_image_bc"]
        rec_image_vb    = outputs["rec_image_vb"]
        map_ref_ex      = outputs["map_ref_ex"]
        map_ref_bc      = outputs["map_ref_bc"]
        map_ref_vb      = outputs["map_ref_vb"]
        #
        loss_unary_ex   = self.loss(rec_image_ex, image_ex)
        loss_unary_bc   = self.loss(rec_image_bc, image_bc)
        loss_unary_vb   = self.loss(rec_image_vb, image_vb)
        loss_pair_ex    = self.loss(map_ref_ex, ref_ex)
        loss_pair_bc    = self.loss(map_ref_bc, ref_bc)
        loss_pair_vb    = self.loss(map_ref_vb, ref_vb)
        loss_unary      = loss_unary_ex + loss_unary_bc + loss_unary_vb
        loss_pair       = loss_pair_ex  + loss_pair_bc  + loss_pair_vb
        loss            = loss_unary    + loss_pair
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        image_ex = datapoint["image_ex"]
        image_bc = datapoint["image_bc"]
        image_vb = datapoint["image_vb"]
        val_ex   = datapoint["val_ex"][0]
        val_bc   = datapoint["val_bc"][0]
        val_vb   = datapoint["val_vb"][0]
        '''
        print(val_ex)
        print(val_bc)
        print(val_vb)
        '''
        (rec_image_ex, rec_image_bc, rec_image_vb, map_ref_ex, map_ref_bc, map_ref_vb) \
            = self.renderer(image_ex, image_bc, image_vb, val_ex, val_bc, val_vb)
        
        return {
            "rec_image_ex": rec_image_ex,
            "rec_image_bc": rec_image_bc,
            "rec_image_vb": rec_image_vb,
            "map_ref_ex"  : map_ref_ex,
            "map_ref_bc"  : map_ref_bc,
            "map_ref_vb"  : map_ref_vb,
        }
    
    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        return None
    
    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        return None


@MODELS.register(name="neurop_re", arch="neurop")
class NeurOP_RE(base.ImageEnhancementModel):
    """Neural Color Operators for Sequential Image Retouching.
    
    References:
        https://github.com/amberwangyili/neurop
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "neurop"
    tasks    : list[Task]   = [Task.LLIE, Task.RETOUCH]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int   = 3,
        out_channels: int   = 3,
        base_nf     : int   = 64,
        encode_nf   : int   = 32,
        pixel_weight: float = 10.0,
        init_weights: Any   = None,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "neurop_re",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        self.fea_dim       = encode_nf * 3
        self.image_encoder = Encoder(in_channels,  encode_nf)
        renderer           = Renderer(in_channels, out_channels, base_nf)
        if init_weights is not None:
            init_weights = core.Path(init_weights)
            if init_weights.is_weights_file():
                state_dict = torch.load(init_weights, weights_only=True)
                state_dict = state_dict["state_dict"]
                for key in list(state_dict.keys()):
                    state_dict[key.replace("renderer.", "")] = state_dict.pop(key)
                renderer.load_state_dict(state_dict)
        
        self.bc_renderer   = renderer.bc_block
        self.bc_predictor  = Predictor(self.fea_dim)
        
        self.ex_renderer   = renderer.ex_block
        self.ex_predictor  = Predictor(self.fea_dim)
        
        self.vb_renderer   = renderer.vb_block
        self.vb_predictor  = Predictor(self.fea_dim)
        
        self.renderers     = [self.bc_renderer,  self.ex_renderer,  self.vb_renderer]
        self.predict_heads = [self.bc_predictor, self.ex_predictor, self.vb_predictor]
        
        # Loss
        self.loss = Loss(pixel_weight=pixel_weight, reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        target   = datapoint.get("ref_image")
        enhanced = outputs.get("enhanced")
        outputs["loss"] = self.loss(enhanced, target)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = x
        b, _, h, w = x.shape
        vals = []
        for nop, predict_head in zip(self.renderers, self.predict_heads):
            img_resized = F.interpolate(input=y, size=(256, int(256 * w / h)), mode="bilinear", align_corners=False)
            feat   = self.image_encoder(img_resized)
            scalar = predict_head(feat)
            vals.append(scalar)
            y = nop(y, scalar)
        y = torch.clamp(y, 0.0, 1.0)
        
        return {
            # "vals"    : vals,
            "enhanced": y
        }
        
# endregion
