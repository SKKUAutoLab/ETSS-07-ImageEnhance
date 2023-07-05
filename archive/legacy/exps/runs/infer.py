#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
import os

from exps import configs
from exps.utils import data_dir
from exps.utils import load_config
from exps.utils import models_zoo_dir
from exps.utils import results_dir
from torchkit.core.runner import Inference
from torchkit.models import MODELS

logger = logging.getLogger()


# MARK: - Main

def main():
    """Main function."""
    # NOTE: Configs
    config  = load_config(config=configs.mprnet_rain.config)
    config2 = load_config(config=configs.mprnet_cityscapes_rain.config)
    ckpt   = os.path.join(
        models_zoo_dir, "mprnet_rain_version_0.ckpt"
    )
    ckpt2 = os.path.join(
        models_zoo_dir, "mprnet_cityscapes_rain_version_0.ckpt"
    )
    infer_data = os.path.join(data_dir, "cam_1_rain.mp4")
    
    # NOTE: Model
    model      = MODELS.build_from_dict(cfg=config.model)
    post_model = MODELS.build_from_dict(cfg=config.model)
    
    # NOTE: Get weights
    model      = model.load_from_checkpoint(checkpoint_path=ckpt, **config.model)
    post_model = post_model.load_from_checkpoint(checkpoint_path=ckpt, **config.model)
    
    # NOTE: Inference
    inference_cfg                  = config.inference
    inference_cfg.default_root_dir = results_dir
    inference                      = Inference(**inference_cfg)
    
    # NOTE: Infer
    inference.run(model=model, post_model=None, data=infer_data)
   

if __name__ == "__main__":
    main()
