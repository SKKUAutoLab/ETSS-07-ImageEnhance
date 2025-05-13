#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
Enhancement," WACV 2022.

References:
    - https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

import os
from typing import Sequence

import torch
import torchvision

import mon
import utils
from modeling import model as mmodel

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # For GPU only

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
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]

    scale_factor = args["network"]["scale_factor"]
    
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
    net = mmodel.enhance_net_nopool(scale_factor, conv_type="dsc").to(device)
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    net.eval()
    
    # Benchmark
    if benchmark:
        h = (512 // scale_factor) * scale_factor
        w = (512 // scale_factor) * scale_factor
        flops, params = mon.compute_efficiency_score(model=net, image_size=(h, w))
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
            image      = utils.image_from_path(str(image_path))
            h0, w0     = mon.image_size(meta["shape"])
            # Scale image to have the resolution of multiple of 4
            image      = utils.scale_image(image, scale_factor, device) if scale_factor != 1 else image
            image      = image.to(device)
            
            # Infer
            timer.tick()
            enhanced, params_maps = net(image)
            timer.tock()
            
            # Post-processing
            enhanced = mon.resize(enhanced, (h0, w0), side=None)
            
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
