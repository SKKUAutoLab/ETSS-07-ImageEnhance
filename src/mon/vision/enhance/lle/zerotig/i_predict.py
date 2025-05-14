#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
Enhancement for Low-Light Images," CVPR 2024.

References:
    - https://github.com/Doyle59217/ZeroIG
"""

import logging
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
import torch.utils
from thop import profile
from torch.autograd import Variable

import mon
from model.model import Finetunemodel

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def save_images(tensor):
    image_numpy = tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    flops, _           = profile(model, inputs=(input_tensor,))
    flops_in_gigaflops = flops / 1e9  # Convert FLOPs to gigaflops (G)
    return flops_in_gigaflops


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
    
    raft_model = mon.ROOT_DIR / args["network"]["raft_model"]
    of_scale   = args["network"]["raft_model"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
        cudnn.enabled   = True
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        logging.info("no gpu device available")
        sys.exit(1)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    is_video = mon.is_video_dataset(data_loader)

    # Model
    model = Finetunemodel(weights, raft_model, of_scale, device).to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        # flops, params = mon.compute_efficiency_score(model=model)
        # mon.console.log(f"FLOPs : {flops:.4f}")
        # mon.console.log(f"Params: {params:.4f}")
        total_params = calculate_model_parameters(model)
        mon.console.log(f"Total Params = {total_params:.4f}")

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
            image      = datapoint["image"]
            
            if resize:
                h0, w0 = mon.image_size(image)
                image  = mon.resize(image, (1080, 1920))
            input = Variable(image).to(device)
            
            # Optimize
            timer.tick()
            enhance, denoise, illum = model(input)
            if not is_video:
                enhance, denoise, illum = model(input)
            timer.tock()
            
            # Post-processing
            if resize:
                enhance = mon.resize(enhance, (h0, w0))
                denoise = mon.resize(denoise, (h0, w0))
            enhance = save_images(enhance)
            denoise = save_images(denoise)
            enhance = cv2.cvtColor(enhance, cv2.COLOR_BGR2RGB)
            denoise = cv2.cvtColor(denoise, cv2.COLOR_BGR2RGB)
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), enhance)
            if save_debug:
                output_dir  = mon.parse_output_dir(save_dir, data_name, f"{mon.SAVE_IMAGE_DIR}_denoise", image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), denoise)
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
