#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Edge-guided Multi-domain RGB-to-TIR image Translation for
Training Vision Tasks with Challenging Labels," ICRA 2023.

References:
    - https://github.com/RPM-Robotics-Lab/sRGB-TIR
"""

import sys

import torch
import torch.optim
import torchvision
from torch.autograd import Variable

import mon
from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import get_config, pytorch03_to_pytorch04

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
    
    opt_path     = str(current_dir / "options" / args["opt_path"])
    opt          = get_config(opt_path)
    
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
    opt["vgg_model_path"] = save_dir
    if args["trainer"] == "MUNIT":
        style_dim   = opt["gen"]["style_dim"]
        style_fixed = Variable(torch.randn(args["num_style"], style_dim, 1, 1).to(device), volatile=False)
        trainer     = MUNIT_Trainer(opt)
    elif args["trainer"] == "UNIT":
        style_dim   = None
        style_fixed = None
        trainer     = UNIT_Trainer(opt)
    else:
        sys.exit("Only support MUNIT|UNIT")
    try:
        state_dict = torch.load(str(weights))
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(str(weights)), args["trainer"])
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    trainer = trainer.to(device).train()
    encode  = trainer.gen_a.encode if args["a2b"] else trainer.gen_b.encode  # encode function
    decode  = trainer.gen_b.decode if args["a2b"] else trainer.gen_a.decode  # decode function
    
    # Benchmark
    if benchmark:
        flops_e, params_e = mon.compute_efficiency_score(model=trainer.gen_a)
        flops_d, params_d = mon.compute_efficiency_score(model=trainer.gen_b)
        flops  = flops_e + flops_d
        params = params_e + params_d
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer = mon.Timer()
    with (mon.create_progress_bar() as pbar):
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            image      = datapoint["image"]
            h0, w0     = mon.image_size(image)
            if resize:
                image = mon.resize(image, imgsz)
            # image = Variable(image.to(device), volatile=True)
            image = image.to(device)
            
            # Infer
            timer.tick()
            if args["trainer"] == "MUNIT":
                content, _ = encode(image)
                if args["synchronized"]:
                    style = style_fixed
                else:
                    style = Variable(torch.randn(args["num_style"], style_dim, 1, 1).to(device), volatile=False)
                for j in range(args["num_style"]):
                    s       = style[j].unsqueeze(0)
                    outputs = decode(content, s)
                    outputs = (outputs + 1) / 2.0
            elif args["trainer"] == "UNIT":
                content, _ = encode(image)
                outputs    = decode(content)
                outputs    = (outputs + 1) / 2.0
            else:
                sys.exit("Only support MUNIT|UNIT")
            timer.tock()
            
            # Post-process
            if resize:
                outputs = mon.resize(outputs, (h0, w0))
            
            # Save
            if save_image:
                if keep_subdirs:
                    rel_path   = image_path.relative_path(data_name)
                    parent_dir = rel_path.parent
                    output_dir = save_dir / rel_path.parents[1] / f"{parent_dir.name}_srgb_tir"
                else:
                    output_dir = save_dir / data_name / "srgb_tir"
                    # output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                # torchvision.utils.save_image(outputs, str(output_path), padding=0, normalize=True)
                torchvision.utils.save_image(outputs, str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
