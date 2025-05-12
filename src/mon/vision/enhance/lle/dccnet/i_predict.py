#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Deep Color Consistent Network for Low Light-Image
Enhancement," CVPR 2022.

References:
    - https://github.com/Ian0926/DCC-Net
"""

import torch
import torch.optim
import torchvision

import mon
import src.model as mmodel

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
    color_net  = mmodel.color_net().to(device)
    # color_net = mon.DataParallel(color_net)
    state_dict     = torch.load(weights, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    color_net.load_state_dict(new_state_dict)
    color_net.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=color_net)
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
            image      = mon.resize(image, divisible_by=32)
            
            # Infer
            timer.tick()
            gray, color_hist, enhanced = color_net(image)
            timer.tock()
            
            # Post-processing
            enhanced = mon.resize(enhanced, (h0, w0))
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                torchvision.utils.save_image(enhanced, str(output_path))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
