#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import torch
import torch.optim
import torchvision

import models
import mon
from utils import make_coord

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    # Model specific
    scale        = args.scale
    scale_max    = args.scale_max
    
    # Model
    model = models.make(torch.load(weights, weights_only=True)["model"], load_sd=True).to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
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
                image      = datapoint.get("image")
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                h      = int(image.shape[-2] * int(scale))
                w      = int(image.shape[-1] * int(scale))
                scale_ = h / image.shape[-2]
                coord  = make_coord((h, w), flatten=False).to(device)
                cell   = torch.ones(1, 2).to(device)
                cell[:, 0] *= 2 / h
                cell[:, 1] *= 2 / w
                cell_factor = max(scale_ / scale_max, 1)
                
                # Infer
                timer.tick()
                pred = model(
                    inp   = ((image - 0.5) / 0.5).to(device),
                    coord = coord.unsqueeze(0),
                    cell  = cell_factor * cell
                )#.squeeze(0)
                pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(1, 3, h, w).cpu()
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(pred, str(output_path))
        
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
