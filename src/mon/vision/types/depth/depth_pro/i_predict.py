#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence

import cv2
import matplotlib
import numpy as np
import torch.optim

import mon
import src.depth_pro as depth_pro

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
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
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]

    config                      = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.patch_encoder_preset = args["network"]["patch_encoder_preset"]
    config.image_encoder_preset = args["network"]["image_encoder_preset"]
    config.decoder_features     = args["network"]["decoder_features"]
    config.use_fov_head         = args["network"]["use_fov_head"]
    config.fov_encoder_preset   = args["network"]["fov_encoder_preset"]
    config.checkpoint_uri       = weights
    format                      = args["network"]["format"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer = mon.Timer()
    cmap  = matplotlib.colormaps.get_cmap("Spectral_r")
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Pre-process
            meta           = datapoint["meta"]
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
                if keep_subdirs:
                    rel_path         = image_path.relative_path(data_name)
                    parent_dir       = rel_path.parent
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
                    "file": gray_save_dir / f"{image_path.stem}.jpg",
                    "data": (depth * 255).astype(np.uint8),
                }
                gray_i  = {
                    "file": gray_i_save_dir / f"{image_path.stem}.jpg",
                    "data": (depth_i * 255).astype(np.uint8),
                }
                color   = {
                    "file": color_save_dir / f"{image_path.stem}.jpg",
                    "data": (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8),
                }
                color_i = {
                    "file": color_i_save_dir / f"{image_path.stem}.jpg",
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
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
