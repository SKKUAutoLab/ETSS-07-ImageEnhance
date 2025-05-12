#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
Enhancement for Low-Light Images," CVPR 2024.

References:
    - https://github.com/Doyle59217/ZeroIG
"""

import logging
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim
import torch.utils
from torch.autograd import Variable
import utils
import mon
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict) -> str:
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

    lr           = args["optimizer"]["lr"]
    weight_decay = args["optimizer"]["weight_decay"]
    
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
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(root, data_dir=args["datamodule"]["root"]),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader
    
    # Model
    model = Network()
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    with mon.create_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            for i, datapoint in enumerate(test_dataloader):
                # Input
                image = datapoint["image"]
                input = Variable(image).to(device)

                optimizer.zero_grad()
                optimizer.param_groups[0]["capturable"] = True
                loss = model._loss(input)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                mon.console.log(f"Epoch {epoch}, Step {i}: {loss}")
            utils.save(model, save_dir / f"{fullname}.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
