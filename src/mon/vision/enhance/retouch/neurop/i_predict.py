#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Neural Color Operators for Sequential Image Retouching,"
ECCV 2022.

References:
    - https://github.com/amberwangyili/neurop
"""

from typing import Sequence

import imageio
import torch

import mon
from models import build_model
from utils import *

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

    opt_path       = str(current_dir / "options" / "test" / args["opt_path"])
    opt            = parse(opt_path)
    opt            = dict_to_nonedict(opt)
    opt["dist"]    = False
    opt["weights"] = weights
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    opt["device"] = device
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model = build_model(opt)
    
    # Benchmark
    if benchmark:
        flops, params = model.compute_efficiency_score()
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
            if resize:
                image = mon.resize(image, imgsz)
            else:
                image = mon.resize(image, divisible_by=32)
            
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
            h1, w1  = mon.image_size(sr_img)
            if h1 != h0 or w1 != w0:
                sr_img = mon.resize(sr_img, (h0, w0))
                
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(str(output_path), (255.0 * sr_img).astype("uint8"))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")
    

# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
