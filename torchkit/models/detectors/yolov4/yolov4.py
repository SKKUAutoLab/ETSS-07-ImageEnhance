#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scaled-YOLOv4 Detectors.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import nn

from torchkit.core.fileio import is_json_file
from torchkit.core.fileio import is_name
from torchkit.core.fileio import is_yaml_file
from torchkit.core.fileio import load
from torchkit.core.image import scale_image
from torchkit.core.layer import Bottleneck
from torchkit.core.layer import BottleneckCSP
from torchkit.core.layer import BottleneckCSP2
from torchkit.core.layer import C3
from torchkit.core.layer import Concat
from torchkit.core.layer import ConvBnMish
from torchkit.core.layer import CrossConv
from torchkit.core.layer import Detect
from torchkit.core.layer import DWConv
from torchkit.core.layer import Focus
from torchkit.core.layer import SPP
from torchkit.core.layer import SPPCSP
from torchkit.core.layer import VoVCSP
from torchkit.core.runner import BaseModel
from torchkit.core.utils import Arrays
from torchkit.core.utils import check_anchor_order
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import Images
from torchkit.core.utils import Indexes
from torchkit.core.utils import make_divisible
from torchkit.core.utils import Tensors
from torchkit.models.builder import DETECTORS
from torchkit.models.builder import MODELS
from torchkit.utils import models_zoo_dir

logger = logging.getLogger()


# MARK: - Common Modules

Conv = ConvBnMish


# MARK: - Yolov4

current_dir = os.path.dirname(os.path.abspath(__file__))

cfgs = {
    "yolov4_csp": os.path.join(current_dir, "yolov4_csp.yaml"),
    "yolov4_p5":  os.path.join(current_dir, "yolov4_p5.yaml"),
    "yolov4_p6":  os.path.join(current_dir, "yolov4_p6.yaml"),
    "yolov4_p7":  os.path.join(current_dir, "yolov4_p7.yaml"),
}


# noinspection PyTypeChecker
@DETECTORS.register(name="yolov4")
@MODELS.register(name="yolov4")
class Yolov4(BaseModel):
    """Base class for all YOLOv4 Detectors.

    Attributes:
        cfg (str, list, dict, optional):
			The config to build the model"s layers.
			- If `str`, use the corresponding config from the predefined
			  config dict. This is used to build the model dynamically.
			- If a file or filepath, it leads to the external config file that
			  is used to buikd the model dynamically.
			- If `list`, then each element in the list is the corresponding
			  config for each layer in the model. This is used to build the
			  model dynamically.
			- If `dict`, it usually contains the hyperparameters used to
			  build the model manually in the code.
			- If `None`, then you should manually define the model.
			Remark: You have 5 ways to build the model, so choose the style
			that you like.
    
    Args:
    	name (str, optional):
			Name of the model. Default: `yolov4`.
		num_classes (int, optional):
			Number of classes for classification. Default: `None`.
		out_indexes (Indexes):
    		The list of layers" indexes to extract features. This is called
    		in `forward_features()` and is useful when the model is used as a
    		component in another model.
    		- If is a `tuple` or `list`, return an array of features.
    		- If is a `int`, return only the feature from that layer"s index.
    		- If is `-1`, return the last layer"s output.
    		Default: `-1`.
		pretrained (bool, str):
			Use pretrained weights. If `True`, returns a model pre-trained on
			ImageNet. If `str`, load weights from saved file. Default: `True`.
			- If `True`, returns a model pre-trained on ImageNet.
			- If `str` and is a weight file(path), then load weights from
			  saved file.
			- In each inherited model, `pretrained` can be a dictionary"s
			  key to get the corresponding local file or url of the weight.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        cfg        : Union[str, list, dict],
        channels   : int                    = 3,
        name       : Optional[str]          = "yolov4",
        num_classes: Optional[int]          = None,
        out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        super().__init__(
            name=name, num_classes=num_classes, out_indexes=out_indexes,
            pretrained=pretrained, *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        cfg = self.parse_cfg(cfg)
        assert isinstance(cfg, dict)
        
        if self.num_classes and self.num_classes != cfg["num_classes"]:
            print("Overriding %s num_classes=%g with num_classes=%g" %
                  (cfg, cfg["num_classes"], self.num_classes))
            cfg["num_classes"] = self.num_classes
        self.cfg = cfg
        
        # NOTE: Model
        self.model, self.save = self.parse_model(self.cfg, channels=[channels])

        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.initialize_detector(channels=channels)
            self.initialize_weights()
    
    # MARK: Configure
    
    @staticmethod
    def parse_cfg(cfg: Union[str, list, dict]) -> dict:
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        if is_name(cfg):
            root, ext = os.path.splitext(cfg)
            dir       = os.path.dirname(os.path.abspath(__file__))
            cfg       = os.path.join(dir, f"{root}.yaml")
        if isinstance(cfg, str) and (is_yaml_file(cfg) or is_json_file(cfg)):
            cfg = load(path=cfg)
        return cfg
    
    @staticmethod
    def parse_model(cfg: dict, channels: list) -> tuple[nn.Sequential, list]:
        # print("\n%3s%18s%3s%10s  %-40s%-30s" %
        #       ("", "from", "n", "params", "module", "arguments"))
        
        anchors        = cfg["anchors"]
        num_classes    = cfg["num_classes"]
        depth_multiple = cfg["depth_multiple"]
        width_multiple = cfg["width_multiple"]
        num_anchors    = ((len(anchors[0]) // 2)
                          if isinstance(anchors, list) else anchors)
        num_outputs    = num_anchors * (num_classes + 5)
        layers         = []             # layers
        save           = []             # savelist
        c2             = channels[-1]   # out_channels
        
        for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):
            # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # Eval strings
            for j, a in enumerate(args):
                try:
                    # Eval strings
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass

            n = max(round(n * depth_multiple), 1) if n > 1 else n  # Depth gain
            if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, Focus, CrossConv,
                     BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
                c1, c2 = channels[f], args[0]
                c2     = (make_divisible(c2 * width_multiple, 8)
                          if c2 != num_outputs else c2)
                args   = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [channels[f]]
            elif m is Concat:
                c2 = sum([channels[-1 if x == -1 else x + 1] for x in f])
            elif m is Detect:
                args.append([channels[x + 1] for x in f])
                if isinstance(args[1], int):  # Number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = channels[f]
            
            m_ = (nn.Sequential(*[m(*args) for _ in range(n)])
                  if n > 1 else m(*args))                   # Module
            t  = str(m)[8:-2].replace("__main__.", "")      # Module type
            np = sum([x.numel() for x in m_.parameters()])  # Number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np
            
            # attach index, 'from' index, type, number params
            # print("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))
            save.extend(x % i for x in ([f] if isinstance(f, int) else f)
                        if x != -1)  # Append to savelist
            layers.append(m_)
            channels.append(c2)
        return nn.Sequential(*layers), sorted(save)
    
    def initialize_detector(self, channels: int):
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([
                s / x.shape[-2]
                for x in self.forward_infer(x=torch.zeros(1, channels, s, s))
            ])  # Forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.initialize_biases()  # Only run once
    
    def initialize_weights(self):
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
                # nn.init.kaiming_normal_(m.weight, mode="fan_out",
                # nonlinearity="relu")
            elif t is nn.BatchNorm2d:
                m.eps      = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True
                
    def initialize_biases(self, cf=None):
        """Initialize biases into Detect(), cf is class frequency."""
        m = self.model[-1]                # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            with torch.no_grad():
                b = mi.bias.view(m.num_anchors, -1)  # conv.bias(255) to (3,85)
                b[:, 4]  += math.log(8 / (640 / s) ** 2)
                # obj (8 objects per 640 image)
                b[:, 5:] += (math.log(0.6 / (m.num_classes - 0.99))
                             if cf is None else torch.log(cf / cf.sum()))  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            
    # MARK: Forward Pass

    def forward_train(
        self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs
    ) -> ForwardXYOutput:
        """Forward pass during training with both `x` and `y` are given.

        Args:
            x (torch.Tensor):
                The input image of shape [B, C, H, W].
            y (torch.Tensor):
                The ground-truth label of each input.

        Returns:
            y_hat (torch.Tensor):
                The final predictions.
            metrics (Metrics, optional):
                - A dictionary with the first key must be the `loss`.
                - `None`, training will skip to the next batch.
        """
        # NOTE: By default, call `forward_infer()` and calculate metrics
        y_hat   = self.forward_infer(x=x, *args, **kwargs)
        metrics = {}
        if self.with_loss:
            metrics["loss"] = self.loss(y_hat, y)
        if self.with_metrics:
            ms      = {m.name: m(y_hat, y) for m in self.metrics}
            metrics = metrics | ms  # NOTE: 3.9+ ONLY
        metrics = metrics if len(metrics) else None
        return y_hat, metrics

    def forward_infer(
        self, x: torch.Tensor, augment: bool = False, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass during inference with only `x` is given.

        Args:
            x (torch.Tensor):
                The input image of shape [B, C, H, W].
            augment (bool):
                Test-time augmentation. Default: `False`.
                
        Returns:
            y_hat (torch.Tensor):
                The final predictions.
        """
        if augment:
            img_size = x.shape[-2:]     # H, W
            s        = [1, 0.83, 0.67]  # Scales
            f        = [None, 3, None]  # Flips (2-ud, 3-lr)
            y_hat    = []  # outputs
            for si, fi in zip(s, f):
                xi           = scale_image(x.flip(fi) if fi else x, si)
                yi           = self.forward_features(xi)[0]  # Forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # De-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # De-flip lr
                y_hat.append(yi)
            return torch.cat(y_hat, 1)  # Augmented inference, train
        else:
            return self.forward_features(x, -1)  # Single-scale inference, train

    def forward_features(
        self, x: torch.Tensor, out_indexes: Optional[Indexes] = None
    ) -> Tensors:
        """Forward pass for features extraction.

        Args:
            x (torch.Tensor):
                The input image.
            out_indexes (Indexes, optional):
                The list of layers" indexes to extract features. This is called
                in `forward_features()` and is useful when the model
                is used as a component in another model.
                - If is a `tuple` or `list`, return an array of features.
                - If is a `int`, return only the feature from that layer"s
                  index.
                - If is `-1`, return the last layer"s output.
                Default: `None`.
        """
        out_indexes = self.out_indexes if out_indexes is None else out_indexes
        
        y     = []
        y_hat = []  # Outputs
        for idx, m in enumerate(self.model):
            if m.f != -1:  # If not from previous layer
                # From earlier layers
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            
            if isinstance(out_indexes, (tuple, list)) and (idx in out_indexes):
                y_hat.append(x)
            elif isinstance(out_indexes, int) and (idx == out_indexes):
                return x
            else:
                y_hat = x
        return y_hat

    # MARK: Visualization

    def show_results(
        self,
        y_hat        : Images,
        x            : Optional[Images] = None,
        y            : Optional[Images] = None,
        filepath     : Optional[str]    = None,
        image_quality: int              = 95,
		show         : bool             = False,
		show_max_n   : int              = 8,
		wait_time    : float            = 0.01,
        *args, **kwargs
    ):
        """Draw `result` over input image.

        Args:
            y_hat (Images):
                The enhanced images.
            x (Images, optional):
                The low quality images.
            y (Images, optional):
                The high quality images.
            filepath (str, optional):
				The file path to save the debug result.
			image_quality (int):
				The image quality to be saved. Default: `95`.
			show (bool):
				If `True` shows the results on the screen. Default: `False`.
			show_max_n (int):
				Maximum debugging items to be shown. Default: `8`.
			wait_time (float):
				Pause some times before showing the next image.
        """
        pass


# MARK: - Yolov4-CSP

@DETECTORS.register(name="yolov4_csp")
@MODELS.register(name="yolov4_csp")
class Yolov4CSP(Yolov4):
    """Yolov4-CSP variant from `Scaled-YOLOv4: Scaling Cross Stage Partial
    Network`.
    """
    
    model_zoo = {
        "coco": dict(
            path=os.path.join(models_zoo_dir, "yolov4_csp_coco.weights"),
            file_name="yolov4_csp_coco.weights", num_classes=80,
        )
    }
    
    def __init__(
        self,
        channels   : int                    = 3,
        name       : Optional[str]          = "yolov4_csp",
        num_classes: Optional[int]          = None,
        out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg         = "yolov4_csp",
            channels    = channels,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - Yolov4-P5

@DETECTORS.register(name="yolov4_p5")
@MODELS.register(name="yolov4_p5")
class Yolov4P5(Yolov4):
    """Yolov4-P5 variant from `Scaled-YOLOv4: Scaling Cross Stage Partial
    Network`.
    """
    
    model_zoo = {
        "coco": dict(
            path=os.path.join(models_zoo_dir, "yolov4_p5_coco.pth"),
            file_name="yolov4_p5_coco.pth", num_classes=80,
        )
    }
    
    def __init__(
        self,
        channels   : int                    = 3,
        name       : Optional[str]          = "yolov4_p5",
        num_classes: Optional[int]          = None,
        out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg         = "yolov4_p5",
            channels    = channels,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - Yolov4-P6

@DETECTORS.register(name="yolov4_p6")
@MODELS.register(name="yolov4_p6")
class Yolov4P6(Yolov4):
    """Yolov4-P6 variant from `Scaled-YOLOv4: Scaling Cross Stage Partial
    Network`.
    """
    
    model_zoo = {
        "coco": dict(
            path=os.path.join(models_zoo_dir, "yolov4_p6_coco.pth"),
            file_name="yolov4_p6_coco.pth", num_classes=80,
        )
    }
    
    def __init__(
        self,
        channels   : int                    = 3,
        name       : Optional[str]          = "yolov4_p6",
        num_classes: Optional[int]          = None,
        out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg         = "yolov4_p6",
            channels    = channels,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - Yolov4-P7

@DETECTORS.register(name="yolov4_p7")
@MODELS.register(name="yolov4_p7")
class Yolov4P7(Yolov4):
    """Yolov4-P7 variant from `Scaled-YOLOv4: Scaling Cross Stage Partial
    Network`.
    """
    
    model_zoo = {
        "coco": dict(
            path=os.path.join(models_zoo_dir, "yolov4_p7_coco.pth"),
            file_name="yolov4_p7_coco.pth", num_classes=80,
        )
    }
    
    def __init__(
        self,
        channels   : int                    = 3,
        name       : Optional[str]          = "yolov4_p7",
        num_classes: Optional[int]          = None,
        out_indexes: Indexes                = -1,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg         = "yolov4_p7",
            channels    = channels,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
