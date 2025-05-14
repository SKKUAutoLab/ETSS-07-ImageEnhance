#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Super-Resolution Neural Operator," CVPR 2023.

References:
    - https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

import torch
import torch.optim
import torchvision

import models
import mon
from utils import make_coord

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
    
    scale        = args["scale"]
    scale_max    = args["scale_max"]
    
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
    model = models.make(torch.load(weights, weights_only=True)["model"], load_sd=True).to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=model)
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
            meta        = datapoint["meta"]
            image_path  = mon.Path(meta["path"])
            image       = datapoint["image"].to(device)
            h           = int(image.shape[-2] * int(scale))
            w           = int(image.shape[-1] * int(scale))
            scale_      = h / image.shape[-2]
            coord       = make_coord((h, w), flatten=False).to(device)
            cell        = torch.ones(1, 2).to(device)
            cell[:, 0] *= 2 / h
            cell[:, 1] *= 2 / w
            cell_factor = max(scale_ / scale_max, 1)
            
            # Infer
            timer.tick()
            pred = model(
                inp   = ((image - 0.5) / 0.5).to(device),
                coord = coord.unsqueeze(0),
                cell  = cell_factor * cell
            )#.squeeze(0)
            pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(1, 3, h, w).cpu()
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(pred, str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
