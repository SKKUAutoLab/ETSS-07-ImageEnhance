#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier
Frequency Information," ACMMM 2023.

References:
    - https://github.com/wangchx67/FourLLIE
"""

import cv2
import numpy as np
import torch

import data.util as dutil
import mon
import options.options as option
import utils.util as util
from models import create_model

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
    
    opt_path = str(current_dir / "options" / "test" / args["opt_path"])
    opt      = option.parse(opt_path, is_train=False)
    opt      = option.dict_to_nonedict(opt)
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    opt["device"] = device
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    opt["path"]["pretrain_model_G"] = str(weights)
    model = create_model(opt)
    
    # Measure efficiency score
    if benchmark:
        flops, params = model.compute_efficiency_score()
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
            image      = dutil.read_img(None, str(image_path))
            image      = image[:, :, ::-1]
            h0, w0     = mon.image_size(image)
            image      = mon.resize(image, divisible_by=32)
            image_nf   = cv2.blur(image, (5, 5))
            image_nf   = image_nf * 1.0 / 255.0
            image_nf   = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
            image      = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
            image      = image.unsqueeze(0).to(device)
            image_nf   = image_nf.unsqueeze(0).to(device)
            
            # Infer
            timer.tick()
            model.feed_data(
                data    = {
                    "idx"   : i,
                    "LQs"   : image,
                    "nf"    : image_nf,
                    "border": 0,
                },
                need_GT = False,
            )
            model.test()
            timer.tock()
            
            # Post-processing
            visuals  = model.get_current_visuals(need_GT=False)
            enhanced = util.tensor2img(visuals["rlt"])  # uint8
            enhanced = cv2.resize(enhanced, (w0, h0))
            
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
