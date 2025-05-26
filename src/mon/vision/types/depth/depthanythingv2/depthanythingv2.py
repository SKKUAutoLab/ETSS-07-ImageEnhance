#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wraps DepthAnythingV2 models for easy loading and inference."""

__all__ = [
    "DepthAnythingV2_ViTB",
    "DepthAnythingV2_ViTL",
    "DepthAnythingV2_ViTS",
    "build_depth_anything_v2",
]

from abc import ABC
from typing import Any, Literal

from depth_anything_v2 import dpt
from mon import core, nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.vision.types.depth import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class DepthAnythingV2(nn.ExtraModel, base.DepthEstimationModel, ABC):
    """This class implements a wrapper for `DepthAnythingV2` models
    defined in `mon_extra.vision.depth.depth_anything_v2`.
    """
    
    arch     : str          = "depth_anything_v2"
    tasks    : list[Task]   = [Task.DEPTH]
    mltypes  : list[MLType] = [MLType.INFERENCE]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        pass
    
    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        x = datapoint["image"]
        y = self.model(x)
        y = (y - y.min()) / (y.max() - y.min())  # Normalize the depth map in the range [0, 1].
        y = y.unsqueeze(1)
        return {"depth": y}


@MODELS.register(name="depth_anything_v2_vits", arch="depth_anything_v2")
class DepthAnythingV2_ViTS(DepthAnythingV2):
    
    name: str = "depth_anything_v2_vits"
    zoo : dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/dtype/depth/depth_anything_v2/depth_anything_v2_vits/da_2k/depth_anything_v2_vits_da_2k.pth",
            "num_classes": None,
        },
    }
    
    def __init__(self, weights: Any = "da_2k", *args, **kwargs):
        super().__init__(weights=weights, *args, **kwargs)
        
        # Network
        self.model = dpt.DepthAnythingV2(
            encoder      = "vits",
            features     = 64,
            out_channels = [48, 96, 192, 384],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="depth_anything_v2_vitb", arch="depth_anything_v2")
class DepthAnythingV2_ViTB(DepthAnythingV2):
    
    name: str = "depth_anything_v2_vitb"
    zoo : dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/dtype/depth/depth_anything_v2/depth_anything_v2_vitb/da_2k/depth_anything_v2_vitb_da_2k.pth",
            "num_classes": None,
        },
    }

    def __init__(self, weights: Any = "da_2k", *args, **kwargs):
        super().__init__(weights=weights, *args, **kwargs)
        
        # Network
        self.model = dpt.DepthAnythingV2(
            encoder      = "vitb",
            features     = 128,
            out_channels = [96, 192, 384, 768],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            

@MODELS.register(name="depth_anything_v2_vitl", arch="depth_anything_v2")
class DepthAnythingV2_ViTL(DepthAnythingV2):
    
    name: str = "depth_anything_v2_vitl"
    zoo : dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/dtype/depth/depth_anything_v2/depth_anything_v2_vitl/da_2k/depth_anything_v2_vitl_da_2k.pth",
            "num_classes": None,
        },
    }

    def __init__(self, weights: Any = "da_2k", *args, **kwargs):
        super().__init__(weights=weights, *args, **kwargs)
        
        # Network
        self.model = dpt.DepthAnythingV2(
            encoder      = "vitl",
            features     = 256,
            out_channels = [256, 512, 1024, 1024],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


def build_depth_anything_v2(
    encoder     : Literal["vits", "vitb", "vitl", "vitg"] = "vits",
    in_channels : int = 3,
    weights     : Any = "da_2k",
    *args, **kwargs
) -> DepthAnythingV2:
    if encoder not in ["vits", "vitb", "vitl", "vitg"]:
        raise ValueError(f"`encoder` must be one of ['vits', 'vitb', 'vitl', 'vitg'], got {encoder}.")
    if encoder == "vits":
        return DepthAnythingV2_ViTS(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitb":
        return DepthAnythingV2_ViTB(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitl":
        return DepthAnythingV2_ViTL(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitg":
        raise NotImplementedError("The `vitg` encoder has been not implemented yet.")
