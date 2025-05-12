#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    https://github.com/pvnieo/Low-light-Image-Enhancement
"""

from typing import Sequence

import cv2

import mon
from exposure_enhancement import enhance_image_exposure

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
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]
    
    gamma   = args["network"]["gamma"]
    lambda_ = args["network"]["lambda_"]
    dual    = not args["network"]["lime"]
    sigma   = args["network"]["sigma"]
    bc      = args["network"]["bc"]
    bs      = args["network"]["bs"]
    be      = args["network"]["be"]
    eps     = args["network"]["eps"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, False, verbose=False)
    
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
            h0, w0     = mon.image_size(image)
            if resize:
                image = cv2.resize(image, (imgsz, imgsz))
            
            # Infer
            timer.tick()
            enhanced = enhance_image_exposure(
                im=image, gamma=gamma, lambda_=lambda_, dual=dual, sigma=sigma, bc=bc,
                bs=bs, be=be, eps=eps
            )
            timer.tock()
            
            # Post-processing
            if resize:
                enhanced = cv2.resize(enhanced, (w0, h0))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                cv2.imwrite(str(output_path), enhanced)
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
