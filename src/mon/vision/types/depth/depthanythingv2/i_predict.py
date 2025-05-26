#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Depth Anything V2. A More Capable Foundation Model for
Monocular Depth Estimation," NeurIPS 2024.

References:
    - https://github.com/DepthAnything/Depth-Anything-V2
"""

from typing import Sequence

import cv2
import matplotlib
import numpy as np
import torch
import torch.optim

import mon
from depth_anything_v2.dpt import DepthAnythingV2

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
    torchrun     = args["torchrun"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    seed         = args["seed"]
    batch_size   = args["batch_size"]
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
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

    encoder      = args["network"]["encoder"]
    features     = args["network"]["features"]
    out_channels = args["network"]["out_channels"]
    pred_only    = args["network"]["pred_only"]
    format       = args["network"]["format"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, False, verbose=False)
    
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
    depth_anything = DepthAnythingV2(encoder=encoder, features=features, out_channels=out_channels, device=device).to(device)
    depth_anything.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    depth_anything = depth_anything.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=depth_anything)
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
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            image      = datapoint["image"]
            
            # Infer
            timer.tick()
            depth = depth_anything.infer_image(image, imgsz)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            timer.tock()
            
            # Save Image
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                mon.save_image(np.repeat(depth[..., np.newaxis], 3, axis=-1), output_path)

            # Save Debug
            if save_debug:
                debug_dir = mon.parse_output_dir(save_dir, data_name, mon.SAVE_DEBUG_DIR, image_path, keep_subdirs, save_nearby)
                if save_nearby:
                    debug_dir = debug_dir.parent / f"{debug_dir.stem}_c"
                debug_path = debug_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                mon.save_image((cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8), debug_path)

                '''
                if keep_subdirs:
                    rel_path       = image_path.relative_path(data_name)
                    parent_dir     = rel_path.parent
                    gray_save_dir  = save_dir / rel_path.parents[1] / f"{parent_dir.name}_dav2_{encoder}_g"
                    color_save_dir = save_dir / rel_path.parents[1] / f"{parent_dir.name}_dav2_{encoder}_c"
                    # gray_save_dir  = save_dir / parent_dir.parent / f"{parent_dir.name}_dav2_{encoder}_g" / image_path.relative_path(relative_path.parent.name)
                    # color_save_dir = save_dir / parent_dir.parent / f"{parent_dir.name}_dav2_{encoder}_c" / image_path.relative_path(relative_path.parent.name)
                else:
                    gray_save_dir  = save_dir / data_name / "gray"
                    color_save_dir = save_dir / data_name / "color"
                gray    = {
                    "file": gray_save_dir / f"{image_path.stem}.jpg",
                    "data": np.repeat(depth[..., np.newaxis], 3, axis=-1),
                }
                color   = {
                    "file": color_save_dir / f"{image_path.stem}.jpg",
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
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if not pred_only:
                        split_region    = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
                        combined_result = cv2.hconcat([image, split_region, output])
                        output          = combined_result
                    cv2.imwrite(str(output_path), output)
                '''
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
