#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "HVI: A New color space for Low-light Image Enhancement,"
CVPR 2025.

References:
    - https://github.com/Fediory/HVI-CIDNet
"""

import cv2
import torch
import torchvision

import mon
from net.CIDNet import CIDNet

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
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]
    
    gated  = args["network"]["gated"]
    gated2 = args["network"]["gated2"]
    alpha  = args["network"]["alpha"]
    gamma  = args["network"]["gamma"]
    
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
    torch.set_grad_enabled(False)
    model = CIDNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))
    model.eval()
    model.trans.gated  = gated
    model.trans.gated2 = gated2
    model.trans.alpha  = alpha
    
    weights = mon.Path(weights)
    if weights.name == "hvi_cidnet_lol_v2_real_w_perc.pth":
        model.trans.alpha = 0.84
    elif weights.name == "hvi_cidnet_lol_v2_real_best_ssim.pth":
        model.trans.alpha = 0.82
    
    # Measure efficiency score
    if benchmark:
        flops, params = mon.compute_efficiency_score(model)
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
            enhanced = model(image ** gamma)
            timer.tock()
            
            # Post-processing
            enhanced = torch.clamp(enhanced, 0, 1)
            enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = torchvision.transforms.ToPILImage()(enhanced.squeeze(0))
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                enhanced.save(str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
