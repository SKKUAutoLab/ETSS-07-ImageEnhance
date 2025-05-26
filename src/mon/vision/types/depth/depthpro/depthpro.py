#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wraps DepthPro models for easy loading and inference."""

__all__ = [
    "DepthPro",
]

from typing import Any

import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.vision.types.depth import base
from .src import depth_pro

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
@MODELS.register(name="depth_pro", arch="depth_pro")
class DepthPro(nn.ExtraModel, base.DepthEstimationModel):
    """This class implements a wrapper for `DepthAnythingV2` models
    defined in `mon_extra.vision.depth.depth_anything_v2`.
    """
    
    arch     : str          = "depth_pro"
    name     : str          = "depth_pro"
    tasks    : list[Task]   = [Task.DEPTH]
    mltypes  : list[MLType] = [MLType.INFERENCE]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "pretrained": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/dtype/depth/depth_pro/depth_pro/pretrained/depth_pro.pt",
            "num_classes": None,
        },
    }
    
    def __init__(
        self,
        patch_encoder_preset: str  = "dinov2l16_384",
        image_encoder_preset: str  = "dinov2l16_384",
        decoder_features    : int  = 256,
        use_fov_head        : bool = True,
        fov_encoder_preset  : str  = "dinov2l16_384",
        weights             : Any  = "pretrained",
        *args, **kwargs
    ):
        super().__init__(weights=weights, *args, **kwargs)
        config                      = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        config.patch_encoder_preset = patch_encoder_preset
        config.image_encoder_preset = image_encoder_preset
        config.decoder_features     = decoder_features
        config.use_fov_head         = use_fov_head
        config.fov_encoder_preset   = fov_encoder_preset
        config.checkpoint_uri       = self.weights
        
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms(config=config)
        self.model.eval()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        pass
    
    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        x       = datapoint["image"]
        f_px    = datapoint["f_px"]
        outputs = self.model.infer(x, f_px=f_px)
        return {
            "focallength_px": outputs["focallength_px"],
            "depth"         : outputs["depth"],
        }
    
    # ----- Predict -----
    def infer(self, datapoint : dict, *args, **kwargs) -> dict:
        # Pre-processing
        meta               = datapoint["meta"]
        image_path         = core.Path(meta["path"])
        image, _, f_px     = depth_pro.load_rgb(str(image_path))
        image              = self.transform(image)
        datapoint["image"] = image
        datapoint["f_px"]  = f_px
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Forward
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint, *args, **kwargs)
        timer.tock()
        
        # Return
        return outputs | {
            "time": timer.avg_time
        }
