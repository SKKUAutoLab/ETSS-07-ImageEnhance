#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.

References:
    - https://github.com/aasharma90/RetinexNet_PyTorch
"""

import mon
from model import RetinexNet

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
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
    model = RetinexNet(imgsz, benchmark).to(device)
    
    # Predicting
    timer = mon.Timer()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Listing images",
        ):
            meta        = datapoint["meta"]
            image_path  = meta["path"]
            output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
            output_dir.mkdir(parents=True, exist_ok=True)
            timer.tick()
            model.predict([image_path], res_dir=str(output_dir), ckpt_dir=str(weights))
            timer.tock()
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
