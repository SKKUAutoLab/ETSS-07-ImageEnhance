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
import src.depth_pro as depth_pro

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
    imgsz        = imgsz[0] if isinstance(imgsz, list | tuple) else imgsz
    resize       = args.resize
    benchmark    = False  # args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    format       = args.format
    
    config                      = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.patch_encoder_preset = args.patch_encoder_preset
    config.image_encoder_preset = args.image_encoder_preset
    config.decoder_features     = args.decoder_features
    config.use_fov_head         = args.use_fov_head
    config.fov_encoder_preset   = args.fov_encoder_preset
    config.checkpoint_uri       = weights
    
    # Model
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
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
    
    # Predicting
    cmap  = matplotlib.colormaps.get_cmap("Spectral_r")
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Pre-process
                meta           = datapoint.get("meta")
                image_path     = mon.Path(meta["path"])
                image, _, f_px = depth_pro.load_rgb(str(image_path))
                image          = transform(image)
                
                # Infer
                timer.tick()
                outputs        = model.infer(image, f_px=f_px)
                depth          = outputs["depth"]
                focallength_px = outputs["focallength_px"]
                timer.tock()
                
                # Post-process
                depth   = depth.detach().cpu().numpy().squeeze()
                depth   = (depth - depth.min()) / (depth.max() - depth.min())
                depth_i = 1.0 - depth
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path         = image_path.relative_path(data_name)
                        parent_dir       = rel_path.parent.parent
                        gray_save_dir    = save_dir / rel_path.parents[1] / f"{parent_dir.name}_depth_pro_g"
                        color_save_dir   = save_dir / rel_path.parents[1] / f"{parent_dir.name}_depth_pro_c"
                        gray_i_save_dir  = save_dir / rel_path.parents[1] / f"{parent_dir.name}_depth_pro_g_i"
                        color_i_save_dir = save_dir / rel_path.parents[1] / f"{parent_dir.name}_depth_pro_c_i"
                    else:
                        gray_save_dir    = save_dir / data_name / "gray"
                        color_save_dir   = save_dir / data_name / "color"
                        gray_i_save_dir  = save_dir / data_name / "gray_i"
                        color_i_save_dir = save_dir / data_name / "color_i"
                    gray    = {
                        "file": gray_save_dir / image_path.name,
                        "data": (depth * 255).astype(np.uint8),
                    }
                    gray_i  = {
                        "file": gray_i_save_dir / image_path.name,
                        "data": (depth_i * 255).astype(np.uint8),
                    }
                    color   = {
                        "file": color_save_dir / image_path.name,
                        "data": (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8),
                    }
                    color_i = {
                        "file": color_i_save_dir / image_path.name,
                        "data": (cmap(depth_i)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8),
                    }
                    results = []
                    if format in [2, "all"]:
                        results = [gray, gray_i, color, color_i]
                    elif format in [0, "gray", "grayscale"]:
                        results = [gray, gray_i]
                    elif format in [1, "color"]:
                        results = [color, color_i]
                    
                    for result in results:
                        output_path = result["file"]
                        output      = result["data"]
                        output_path.parent.mkdir(parents=True, exist_ok=True)
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
