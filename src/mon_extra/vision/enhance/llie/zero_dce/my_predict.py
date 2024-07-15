#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model as mmodel
import mon

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    DCE_net = mmodel.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load(weights))
    DCE_net.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(DCE_net),
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
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path    = meta["path"]
                data_lowlight = Image.open(image_path).convert("RGB")
                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                data_lowlight = data_lowlight.permute(2, 0, 1)
                data_lowlight = data_lowlight.to(device).unsqueeze(0)
                start_time    = time.time()
                _, enhanced_image, _ = DCE_net(data_lowlight)
                run_time      = (time.time() - start_time)
                output_path   = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion