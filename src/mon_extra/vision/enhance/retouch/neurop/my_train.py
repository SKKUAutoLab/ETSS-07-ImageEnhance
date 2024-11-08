#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeurOP.

This module implements the paper: "Neural Color Operators for Sequential Image
Retouching".

References:
    https://github.com/amberwangyili/neurop
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

import mon
from data import build_train_loader
from models import build_model
from utils import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Train

def train(args: argparse.Namespace):
    # General config
    opt_path = str(current_dir / "model_config" / "train" / args.opt_path)
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    device   = mon.set_device(args.device)
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Override options with args
    opt = parse(opt_path)
    opt = dict_to_nonedict(opt)
    opt["network_G"]["init_model"] = mon.ZOO_DIR / opt["network_G"]["init_model"]
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model = build_model(opt)
    
    # Data I/O
    dataset_opt  = opt["datasets"]
    train_loader = build_train_loader(dataset_opt)
    
    # Seed
    seed = opt["train"]["manual_seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_iters  = opt["train"]["niter"]
    total_epochs = int(total_iters / len(train_loader))
    
    # Training
    current_step = 0
    start_epoch  = 0
    with mon.get_progress_bar() as pbar:
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
            for k,v in logs.items():
                v /= len(train_loader)
                message += "{:s}: {:.4e} ".format(k, v)
            model.log_dict = defaultdict(int)
            model.save("latest", save_dir=weights_dir)
            
# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
