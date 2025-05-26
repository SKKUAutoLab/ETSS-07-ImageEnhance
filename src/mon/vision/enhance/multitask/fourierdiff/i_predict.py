#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint
Low-Light Enhancement and Deblurring," CVPR 2024.

References:
    - https://github.com/aipixel/FourierDiff
"""

import argparse

import torch
import yaml

import mon
from guided_diffusion.diffusion_llie_modified import Diffusion

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
    batch_size   = args["batch_size"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    save_nearby  = args["save_nearby"]
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
    runner = Diffusion(dict2namespace(args), config)

    # Benchmark
    # if benchmark:
    #     flops, params = mon.compute_efficiency_score(model=model, image_size=256, channels=3)
    #     mon.console.log(f"FLOPs : {flops:.4f}")
    #     mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer = mon.Timer()
    # Infer
    timer.tick()
    runner.sample(weights, data_name, data_loader, imgsz, resize, save_image, save_dir, keep_subdirs, save_nearby)
    timer.tock()

    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
