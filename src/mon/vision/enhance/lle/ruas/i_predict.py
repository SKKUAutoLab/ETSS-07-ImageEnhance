#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
Search for Low-light Image Enhancement," 2021.

References:
    - https://github.com/KarelZhang/RUAS
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import mon
from model import Network

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


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
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    cudnn.benchmark = True
    cudnn.enabled   = True
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model = Network().to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
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
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            image      = datapoint["image"].to(device)
            
            # Infer
            timer.tick()
            u_list, r_list = model(image)
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                save_images(u_list[-1], str(output_path))
                # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                """
                if args.model == "lol":
                    save_images(u_list[-1], u_path)
                elif args.model == "upe" or args.model == "dark":
                    save_images(u_list[-2], u_path)
                """
                
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
