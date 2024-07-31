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

console       = mon.console
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
    grayscale    = args.grayscale
    pred_only    = args.pred_only
    
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
    depth_anything.load_state_dict(torch.load(str(weights), map_location="cpu"))
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
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    cmap  = matplotlib.colormaps.get_cmap("Spectral_r")
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path = meta["path"]
                raw_image  = cv2.imread(str(image_path))
                timer.tick()
                depth      = depth_anything.infer_image(raw_image, imgsz)
                depth      = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth      = depth.astype(np.uint8)
                timer.tock()
                
                if grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                if pred_only:
                    output = depth
                else:
                    split_region    = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                    combined_result = cv2.hconcat([raw_image, split_region, depth])
                    output          = combined_result
                output_path = save_dir / image_path.name
                cv2.imwrite(str(output_path), output)
                
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
