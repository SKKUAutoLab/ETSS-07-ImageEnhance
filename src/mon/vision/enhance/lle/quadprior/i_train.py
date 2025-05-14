#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
Priors," CVPR 2024.

References:
    - https://github.com/daooshee/QuadPrior
"""

from cldm.hack import disable_verbosity

disable_verbosity()

import pytorch_lightning as pl
import webdataset as wds
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

import mon
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from coco_dataset import create_webdataset

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    torchrun     = args["torchrun"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    save_nearby  = args["save_nearby"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    sd_locked        = args["network"]["sd_locked"]
    only_mid_control = args["network"]["only_mid_control"]
    lr               = args["optimizer"]["lr"]
    batch_size       = args["datamodule"]["batch_size"]
    logger_freq      = args["logger_freq"]
    
    config_path     = current_dir / args["config_path"]  # "./models/cldm_v15.yaml"
    init_ckpt       = mon.ZOO_DIR / "vision/enhance/lle/quadprior/quadprior/coco/control_sd15_init.ckpt"
    pretrained_ckpt = mon.ZOO_DIR / "vision/enhance/lle/quadprior/quadprior/coco/control_sd15_coco_final.ckpt"
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.parse_device(device)
    device = mon.to_int_list(device) if "auto" not in device else device
    
    # Seed
    mon.set_random_seed(seed)

    # Data I/O
    data       = mon.parse_data_dir(root, data_dir=args["datamodule"]["root"])
    dataset    = create_webdataset(data_dir=str(data))
    dataloader = wds.WebLoader(
        dataset         = dataset,
        batch_size      = batch_size,
        num_workers     = 2,
        pin_memory      = False,
        prefetch_factor = 2,
    )
    
    # Model
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model          = create_model(config_path=config_path).cpu()
    state_dict     = load_state_dict(str(init_ckpt), location="cpu")
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)
    model.add_new_layers()
    
    if pretrained_ckpt != "":
        state_dict = load_state_dict(str(pretrained_ckpt), location="cpu")
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if "_forward_module.control_model" in sd_name:
            new_state_dict[sd_name.replace("_forward_module.control_model.", "")] = sd_param
    model.control_model.load_state_dict(new_state_dict)
    
    model.learning_rate    = lr
    model.sd_locked        = sd_locked
    model.only_mid_control = only_mid_control
    
    # Callback
    logger = ImageLogger(save_dir=str(save_dir), batch_frequency=logger_freq)
    checkpoint_callback = ModelCheckpoint(
        dirpath                 = str(save_dir),
        filename                = fullname + "-{epoch:02d}-{step}",
        # filename                = fullname,
        monitor                 = "step",
        save_last               = False,
        save_top_k              = -1,
        verbose                 = True,
        every_n_train_steps     = 10000,  # How frequent to save checkpoint
        save_on_train_epoch_end = True,
    )
    
    # Trainer
    strategy = DeepSpeedStrategy(
        stage             = 2,
        offload_optimizer = True,
        cpu_checkpointing = True
    )
    trainer = pl.Trainer(
        default_root_dir = str(save_dir),
        devices          = device,
        strategy         = "auto",  # strategy,
        # max_epochs       = epochs,
        max_steps        = steps,
        precision        = 16,
        sync_batchnorm   = True,
        accelerator      = "gpu",
        callbacks        = [logger, checkpoint_callback],
    )
    
    # Train
    trainer.fit(model, dataloader)


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
