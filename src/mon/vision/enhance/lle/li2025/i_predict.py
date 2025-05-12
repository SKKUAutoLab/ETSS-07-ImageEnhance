#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
Real-World low-light Scenarios," ICLR 2025.

References:
    - https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
"""

import torch.optim
import torchvision

import mon
from net.lformer import net
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
    model = net().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
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
            L, _, R, X, I = model(image)
            D = image - X
            I = torch.clamp(I, 0, 1)
            R = torch.clamp(R, 0, 1)
            L = torch.clamp(L, 0, 1)
            timer.tock()
            
            # Post-process
            L = L.cpu()
            R = R.cpu()
            I = I.cpu()
            D = D.cpu()
            # L_img = transforms.ToPILImage()(L.squeeze(0))
            # R_img = transforms.ToPILImage()(R.squeeze(0))
            # I_img = transforms.ToPILImage()(I.squeeze(0))
            # D_img = transforms.ToPILImage()(D.squeeze(0))
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                torchvision.utils.save_image(I, str(output_path))
            
            # Save Debug
            if save_debug:
                debug_dir = mon.parse_debug_dir(save_dir, data_name, image_path, keep_subdirs)
                debug_dir.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(L, str(debug_dir / f"{image_path.stem}_L{mon.SAVE_IMAGE_EXT}"))
                torchvision.utils.save_image(R, str(debug_dir / f"{image_path.stem}_R{mon.SAVE_IMAGE_EXT}"))
                torchvision.utils.save_image(D, str(debug_dir / f"{image_path.stem}_D{mon.SAVE_IMAGE_EXT}"))
            
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
