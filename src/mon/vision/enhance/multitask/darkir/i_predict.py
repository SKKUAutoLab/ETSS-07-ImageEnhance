#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.

References:
    - https://github.com/cidautai/DarkIR
"""

import torch.optim
import torchvision
from ptflops import get_model_complexity_info

import mon
from archs import *
from utils.test_utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def load_model(model, path_weights):
    checkpoints = torch.load(str(path_weights), map_location="cpu", weights_only=False)
    weights     = checkpoints["params"]
    # weights     = {"module." + key: value for key, value in weights.items()}
    model.load_state_dict(weights)
    # print("Loaded weights correctly")
    return model


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
    model, _, _ = create_model(args["network"], rank=0, device=device, torchrun=torchrun)
    model       = load_model(model, path_weights=weights)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params  = mon.compute_efficiency_score(model=model)
        # macs , params2 = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False, verbose=False)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
        # mon.console.log(f"MACs  : {macs:.4f}")
    
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

            if resize and (h0 >= 1500 or w0 >= 1500):
                new_size   = [int(dim // 2) for dim in (h0, w0)]
                downsample = torchvision.transforms.Resize(new_size)
            else:
                downsample = torch.nn.Identity()
            image = downsample(image)

            # Infer
            timer.tick()
            enhanced = model(image, side_loss=False)
            timer.tock()

            # Post-processing
            if resize:
                upsample = torchvision.transforms.Resize((h0, w0))
            else:
                upsample = torch.nn.Identity()
            enhanced = upsample(enhanced)
            enhanced = torch.clamp(enhanced, 0.0, 1.0)
            enhanced = enhanced[:, :, :h0, :w0]

            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
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
