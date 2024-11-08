#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeurOP.

This module implements the paper: "Neural Color Operators for Sequential Image
Retouching".

References:
    https://github.com/amberwangyili/neurop
"""

from __future__ import annotations

import argparse

import imageio
import torch

import mon
from models import build_model
from utils import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    opt_path     = str(current_dir / "model_config" / "test" / args.opt_path)
    data         = args.data
    save_dir     = mon.Path(args.save_dir)
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    imgsz        = imgsz[0] if isinstance(imgsz, list | tuple) else imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Override options with args
    opt            = parse(opt_path)
    opt            = dict_to_nonedict(opt)
    opt["dist"]    = False
    opt["device"]  = device
    opt["weights"] = weights
    
    # Model
    model = build_model(opt)
    
    # Measure efficiency score
    if benchmark:
        flops, params, avg_time = model.measure_efficiency_score(image_size=imgsz)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint.get("image").to(device)
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize(image, divisible_by=32)
                
                val_data = {
                    "LQ": image,
                    "GT": image,
                }
                
                # Infer
                timer.tick()
                model.feed_data(data = {
                    "LQ": image,
                    "GT": image,
                })
                model.test()
                timer.tock()
                
                # Post-processing
                visuals = model.get_current_visuals()
                sr_img  = visuals["rlt"]
                h1, w1  = mon.get_image_size(sr_img)
                if h1 != h0 or w1 != w0:
                    sr_img = mon.resize(sr_img, (h0, w0))
                    
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(str(output_path), (255.0 * sr_img).astype("uint8"))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
        
# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
