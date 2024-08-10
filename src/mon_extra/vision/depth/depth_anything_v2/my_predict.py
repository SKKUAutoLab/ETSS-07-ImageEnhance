#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import cv2
import matplotlib
import numpy as np
import torch
import torch.optim

import mon
from depth_anything_v2.dpt import DepthAnythingV2

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
    imgsz        = args.imgsz[0]
    resize       = args.resize
    benchmark    = args.benchmark
    encoder      = args.encoder
    features     = args.features
    out_channels = args.out_channels
    pred_only    = args.pred_only
    format       = args.format
    
    # Model
    '''
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,   96,   192,  384 ]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,   192,  384,  768 ]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256,  512,  1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    '''
    depth_anything = DepthAnythingV2(encoder=encoder, features=features, out_channels=out_channels)
    depth_anything.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    depth_anything = depth_anything.to(device).eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(depth_anything),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    save_dir       = save_dir / data_name
    gray_save_dir  = save_dir / "gray"
    color_save_dir = save_dir / "color"
    save_dir.mkdir(parents=True, exist_ok=True)
    gray_save_dir.mkdir(parents=True, exist_ok=True)
    color_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    cmap  = matplotlib.colormaps.get_cmap("Spectral_r")
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for image, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path = meta["path"]
                # raw_image  = cv2.imread(str(image_path))
                raw_image  = image
                timer.tick()
                depth      = depth_anything.infer_image(raw_image, imgsz)
                depth      = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth      = depth.astype(np.uint8)
                timer.tock()
                
                gray    = {
                    "file": gray_save_dir / image_path.name,
                    "data": np.repeat(depth[..., np.newaxis], 3, axis=-1),
                }
                color   = {
                    "file": color_save_dir / image_path.name,
                    "data": (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8),
                }
                results = []
                if format in [2, "all"]:
                    results = [gray, color]
                elif format in [0, "gray", "grayscale"]:
                    results = [gray]
                elif format in [1, "color"]:
                    results = [color]
                
                for result in results:
                    output_path = result["file"]
                    output      = result["data"]
                    if not pred_only:
                        split_region    = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                        combined_result = cv2.hconcat([raw_image, split_region, output])
                        output          = combined_result
                    cv2.imwrite(str(output_path), output)
                
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
