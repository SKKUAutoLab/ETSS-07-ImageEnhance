#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DEIM: DETR with Improved Matching for Fast
Convergence," CVPR 2025.

References:
    - https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS
from pprint import pprint
import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TORCH_USE_CUDA_DSA"]   = "1"
# torch.multiprocessing.set_sharing_strategy("file_system")


# ----- Train -----
debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def train(args: dict) -> str:
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
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]

    # Start
    if safe_get_rank() == 0:
        mon.console.rule(f"[bold red] {fullname}")
        mon.console.log(f"Machine: {hostname}")

    # Device
    device = mon.set_device(device)

    # Seed
    mon.set_random_seed(seed)

    # Trainer
    resume = mon.parse_weights_file(root, args["resume"]) if args["resume"] else None
    tuning = mon.parse_weights_file(root, args["tuning"]) if args["tuning"] else None
    if weights and weights.is_weights_file(exist=True):
        resume = weights
        tuning = None
    elif resume and resume.is_weights_file(exist=True):
        tuning = None
    else:
        resume = None
    assert not all([tuning, resume]), "Only support from scratch or resume or tuning at one time."
    use_amp      = args["use_amp"]
    test_only    = args["test_only"]
    print_method = args["print_method"]
    print_rank   = args["print_rank"]

    dist_utils.setup_distributed(print_rank, print_method, seed=seed)

    cfg_path     = current_dir / "options" / args["cfg_path"]
    update_dict  = {"tuning": str(tuning)} if tuning       else {}
    update_dict  = {"resume": str(resume)} if resume       else update_dict
    update_dict |= {"device": device}      if not torchrun else {}
    update_dict |= {
        "seed"        : seed,
        "use_amp"     : use_amp,
        "output_dir"  : str(save_dir),
        "summary_dir" : str(save_dir),
        "test_only"   : test_only,
        "print_method": print_method,
        "print_rank"  : print_rank,
        "epoches"     : epochs,  # Don't know why they use "epoches" instead of "epochs"?
        "__include__" : args.get("__include__", None),
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(root), **update_dict)

    if resume or tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    # Training
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
