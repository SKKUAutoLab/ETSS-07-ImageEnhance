#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "RSFNet: Specularity Factorization for Low Light Enhancement,"
CVPR 2024.

References:
    - https://github.com/sophont01/RSFNet
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import mon
from libs.src.model import RRNet

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=FutureWarning)
eps = np.finfo(np.float32).eps

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


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
    
    factors      = args["network"]["factors"]
    freeze       = args["network"]["freeze"]
    f_over_exp   = args["network"]["f_over_exp"]
    lr           = args["optimizer"]["lr"]
    lr_decay     = args["optimizer"]["lr_decay"]
    max_norm     = args["trainer"]["max_norm"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)

    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(root, data_dir=args["datamodule"]["root"]),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    
    # Model
    model = RRNet(mode="train", device=device, **args["network"]).to(device)
    if weights is not None and mon.Path(weights).is_weights_file():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.apply(weights_init)
    
    # Optimizer
    optimizer = torch.optim.SGD([
        {"params": model.fuseNet.encoder.parameters(), "lr": lr},
        {"params": model.fuseNet.decoder.parameters(), "lr": lr},
    ])
    for i in range(factors):
        optimizer.add_param_group({"params": model.factNet.lmbda_A[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.lmbda_E[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.step[i].parameters(),    "lr": 0.01})  # 0.01
        
    # Training
    for epoch in range(epochs):
        losses = {"train_loss": 0, "L_color": 0, "L_exp": 0, "L_TV": 0, "L_fact": 0}
        model.train()
        model.factNet.et_mean = [[] for _ in range(factors)]
        model.L = dict.fromkeys(("L_color", "L_exp", "L_TV", "L_fact"))
        
        if epoch > freeze + 25:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * lr_decay
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] * lr_decay
            
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(train_dataloader),
                total       = len(train_dataloader),
                description = f"[bright_yellow] Training"
            ):
                image = datapoint["image"].type(torch.float32).to(device)
                
                optimizer.zero_grad()
                enhanced, loss = model(image)
                if f_over_exp:
                    enhanced = 1 - enhanced
                
                losses["train_loss"] += loss.item()
                losses["L_color"]    += model.L["L_color"]
                losses["L_exp"]      += model.L["L_exp"]
                losses["L_TV"]       += model.L["L_TV"]
                losses["L_fact"]     += model.L["L_fact"]
                
                model.freeze_fact(epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # LOL-v1, LOL-v2-real, LOL-v2-synthetic
                optimizer.step()
                
                del enhanced, loss
        
        # Log
        losses["train_loss"] /= len(train_dataloader)
        losses["L_color"]    /= len(train_dataloader)
        losses["L_exp"]      /= len(train_dataloader)
        losses["L_TV"]       /= len(train_dataloader)
        losses["L_fact"]     /= len(train_dataloader)
        print(f"epoch={epoch}"
              f"\ttrain_loss={losses["train_loss"]:0.9f}"
              f"\tL_color={losses["L_color"]:0.9f}"
              f"\tL_exp={losses["L_exp"]:0.9f}"
              f"\tL_TV={losses["L_TV"]:0.9f}"
              f"\tL_fact={losses["L_fact"]:0.9f}")
            
        # Save
        torch.save(model.state_dict(), save_dir / f"{fullname}.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
