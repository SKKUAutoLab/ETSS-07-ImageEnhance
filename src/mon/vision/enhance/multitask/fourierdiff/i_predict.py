#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "LightenDiffusion: Unsupervised Low-Light Image
Enhancement with Latent-Retinex Diffusion Models," ECCV 2024.

References:
    - https://github.com/JianghaiSCU/LightenDiffusion
"""

import argparse

import numpy as np
import torch
import torchvision
import yaml

import mon
from mon.nn import functional as F
import argparse
import logging
import os
import shutil
import sys
import traceback

import numpy as np
import torch
import yaml

from guided_diffusion.diffusion_llie import Diffusion

torch.set_printoptions(sci_mode=False)

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


@torch.no_grad()
def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]

    opt_path     = current_dir / "options" / args["opt_path"]
    with open(str(opt_path), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    config.device = device

    # Seed
    mon.set_random_seed(seed)

    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model_args = argparse.Namespace(**{
        "mode"        : "evaluation",
        "resume"      : str(weights),
        "image_folder": str(save_dir),
    })
    diffusion = DenoisingDiffusion(model_args, config)
    if weights.is_weights_file(exist=True):
        diffusion.load_ddm_ckpt(str(weights), ema=False)
    else:
        mon.console.log(f"Pre-trained model path is missing!")
    diffusion.model.eval()

    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=diffusion.model, channels=6)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer = mon.Timer()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            image      = datapoint["image"].to(device)

            h0, w0     = mon.image_size(image)
            img_h_64   = int(64 * np.ceil(h0 / 64.0))
            img_w_64   = int(64 * np.ceil(w0 / 64.0))
            x_cond     = F.pad(image, (0, img_w_64 - w0, 0, img_h_64 - h0), 'reflect')

            # Infer
            timer.tick()
            enhanced = diffusion.model(torch.cat((x_cond, x_cond), dim=1))["pred_x"]
            enhanced = enhanced[:, :, :h0, :w0]
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_path = output_dir / "predict" / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(enhanced, str(output_path))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
