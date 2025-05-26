#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Neural Color Operators for Sequential Image Retouching,"
ECCV 2022.

References:
    - https://github.com/amberwangyili/neurop
"""

import argparse
from collections import defaultdict

import torch

import mon
from data import build_train_loader
from models import build_model
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: argparse.Namespace):
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
    batch_size   = args["batch_size"]
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
    
    opt_path = current_dir / "options" / "train" / args["opt_path"]
    opt      = parse(str(opt_path))
    opt      = dict_to_nonedict(opt)
    opt["network_G"]["init_model"] = mon.ROOT_DIR / opt["network_G"]["init_model"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    seed = opt["train"]["manual_seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Data I/O
    dataset_opt  = opt["datasets"]
    train_loader = build_train_loader(dataset_opt)
    
    # Model
    model = build_model(opt)
    
    # Training
    current_step = 0
    total_iters  = opt["train"]["niter"]
    total_epochs = int(total_iters / len(train_loader))
    with mon.create_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(total_epochs + 1),
            total       = total_epochs + 1,
            description = f"[bright_yellow] Training"
        ):
            for _, train_data in enumerate(train_loader):
                # print(f"{train_data["LQ_path"]} | {train_data["GT_path"]}")
                current_step += 1
                if current_step > total_iters:
                    break
                model.feed_data(train_data)
                model.optimize_parameters()
            
            # Log
            logs    = model.get_current_log()
            message = "[epoch:{:3d}, iter:{:8,d}, ".format(epoch, current_step)
            for k, v in logs.items():
                v /= len(train_loader)
                message += "{:s}: {:.4e} ".format(k, v)
            model.log_dict = defaultdict(int)
            
            # Save
            model.save("latest", save_dir=save_dir)
            

# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
