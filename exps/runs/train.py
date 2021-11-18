#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
import os
import socket
from copy import deepcopy

from munch import Munch
from pytorch_lightning.loggers import TensorBoardLogger

from exps import configs
from exps.utils import copy_config_file
from exps.utils import data_dir
from exps.utils import datasets_dir
from exps.utils import load_config
from torchkit.core.dataset import DataModule
from torchkit.core.runner import CheckpointCallback
from torchkit.core.runner import get_epoch
from torchkit.core.runner import get_global_step
from torchkit.core.runner import get_latest_checkpoint
from torchkit.core.runner import Inference
from torchkit.core.runner import Phase
from torchkit.core.runner import Trainer
from torchkit.datasets.builder import DATAMODULES
from torchkit.models.builder import ENHANCERS
from torchkit.models.enhancers import End2EndEnhancer

logger = logging.getLogger()


# MARK: - Hosts

hosts = {
    "default":
        Munch(
            phase      = Phase.TRAINING,
            strategy   = "dp",
            gpus       = [0],
            infer_data = os.path.join(data_dir, "cam_1_rain.mp4"),
            config     = configs.mprnet_rain
        ),
}


# MARK: - Main

def main():
    """Main function."""
    # NOTE: Host
    # hostname = socket.gethostname().lower()
    host = hosts["default"]
    
    # NOTE: Configs
    config                  = load_config(config=host.config.config)
    config.trainer.strategy = host.strategy
    config.trainer.gpus     = host.gpus
    
    # NOTE: Data
    dm = DATAMODULES.build_from_dict(cfg=config.data)
    dm.prepare_data()

    # NOTE: Model
    config.model.classlabels = dm.classlabels
    model: End2EndEnhancer = ENHANCERS.build_from_dict(cfg=config.model)

    # NOTE: Training
    if host.phase is Phase.TRAINING:
        copy_config_file(host.config.__file__, model.version_dir)
        dm.setup(phase=host.phase)
        model.phase = host.phase
        train(model=model, dm=dm, config=config)
        
    # NOTE: Testing
    if host.phase is Phase.TESTING:
        dm.setup(phase=host.phase)
        model.phase = host.phase
        test(model=model, dm=dm, config=config)

    # NOTE: Inference
    if host.phase is Phase.INFERENCE:
        model.phase = host.phase
        infer(model=model, data=host.infer_data, config=config)


# MARK: - Training

def train(
    model: End2EndEnhancer, dm: DataModule, config: Munch
) -> End2EndEnhancer:
    """Train the model.
    
    Args:
        model (End2EndEnhancer):
            The model.
        dm (DataModule):
            The datamodule.
        config (Munch):
            The config dictionary.

    Returns:
        model (MBLLEN):
            The trained model.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*last*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)
    
    # NOTE: Checkpoint Callback
    ckpt_callback = CheckpointCallback(**_cfg.checkpoint)
    
    # NOTE: Logger
    tb_logger     = TensorBoardLogger(**_cfg.tb_logger)
    
    # NOTE: Trainer
    trainer_cfg                      = _cfg.trainer
    trainer_cfg.default_root_dir     = model.version_dir
    trainer_cfg.callbacks            = [ckpt_callback]
    trainer_cfg.enable_checkpointing = True
    trainer_cfg.logger               = tb_logger
    
    trainer               = Trainer(**trainer_cfg)
    trainer.current_epoch = get_epoch(ckpt=ckpt)
    trainer.global_step   = get_global_step(ckpt=ckpt)
    
    # NOTE: Train
    trainer.fit(
        model             = model,
        train_dataloaders = dm.train_dataloader,
        val_dataloaders   = dm.val_dataloader,
        ckpt_path         = ckpt,
    )
    
    return model


# MARK: - Testing

def test(model: End2EndEnhancer, dm: DataModule, config: Munch):
    """Test the model.
    
    Args:
        model (End2EndEnhancer):
            The model.
        dm (DataModule):
            The datamodule.
        config (Munch):
            The config dictionary.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*best*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)
    
    # NOTE: Trainer
    trainer_cfg                  = _cfg.trainer
    trainer_cfg.default_root_dir = model.version_dir
    
    trainer               = Trainer(**trainer_cfg)
    trainer.current_epoch = get_epoch(ckpt=ckpt)
    trainer.global_step   = get_global_step(ckpt=ckpt)
    
    # NOTE: Test
    trainer.test(model=model, dataloaders=dm.test_dataloader, ckpt_path=ckpt)


# MARK: - Inference

def infer(model: End2EndEnhancer, data: str, config: Munch):
    """Inference.
    
    Args:
        model (End2EndEnhancer):
            The model.
        data (str):
            The data source.
        config (Munch):
            The config dictionary.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*best*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)
    if ckpt:
        model = model.load_from_checkpoint(checkpoint_path=ckpt, **_cfg.model)
 
    # NOTE: Inference
    inference_cfg                  = _cfg.inference
    inference_cfg.default_root_dir = os.path.join(model.version_dir, "infer")
    
    inference = Inference(**inference_cfg)
    
    # NOTE: Infer
    inference.run(model=model, data=data)
    

if __name__ == "__main__":
    main()
