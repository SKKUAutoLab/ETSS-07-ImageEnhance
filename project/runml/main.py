#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements main running pipeline."""

import os
import subprocess
import sys

import torch

import mon
import menu_rich

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def run_train(args: dict):
    # Get user input
    root         = mon.Path(args["root"])
    task         = args["task"]
    mode         = args["mode"]
    arch         = args["arch"]
    model        = args["model"]
    config       = args["config"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    torchrun     = args["torchrun"]
    master_port  = args["master_port"]
    master_addr  = args["master_addr"]
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

    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model_root      = mon.parse_model_dir(arch, model)
    model           = mon.parse_model_name(model)
    fullname        = fullname if fullname not in [None, "None", ""] else config.stem
    config          = mon.parse_config_file(
        config       = config,
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
    )
    assert config not in [None, "None", ""]
    weights = mon.to_str(weights, ",")

    kwargs  = {}
    flags   = []
    kwargs |= {"--root"    : str(root)}
    kwargs |= {"--arch"    : arch}
    kwargs |= {"--model"   : model}
    kwargs |= {"--config"  : config}
    # kwargs |= {"--data"    : data}
    kwargs |= {"--fullname": fullname}
    kwargs |= {"--save-dir": str(save_dir)}
    kwargs |= {"--weights" : weights}
    kwargs |= {"--device"  : device}
    flags  += ["--torchrun"]     if torchrun     else []
    kwargs |= {"--epochs"  : epochs}
    kwargs |= {"--steps"   : steps}
    kwargs |= {"--seed"    : seed}
    # kwargs |= {"--imgsz"   : imgsz}
    # flags  += ["--resize"]       if resize       else []
    flags  += ["--benchmark"]    if benchmark    else []
    flags  += ["--save-image"]   if save_image   else []
    flags  += ["--save-debug"]   if save_debug   else []
    flags  += ["--use-fullname"] if use_fullname else []
    flags  += ["--keep-subdirs"] if keep_subdirs else []
    flags  += ["--exist-ok"]     if exist_ok     else []
    flags  += ["--verbose"]      if verbose      else []

    # Parse script file
    python_call = ["python"]
    env         = {**os.environ}
    if use_extra_model:
        script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "i_train.py"
        if torchrun:
            device_     = mon.parse_device(device)
            python_call = [
                "python", "-m", "torch.distributed.run",
                f"--nproc_per_node={len(device_)}",
                f"--master_port={master_port}",
                f"--master_addr={master_addr}",
            ]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(device_), **env}
    else:
        script_file = current_dir / "train.py"

    # Parse arguments
    args_call: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        elif isinstance(v, list | tuple):
            args_call_ = [f"{k}={v_}" for v_ in v]
        else:
            args_call_ = [f"{k}={v}"]
        args_call += args_call_
    
    # Run training
    if script_file.is_py_file():
        print("\n")
        command = (
            python_call +
            [script_file] +
            args_call +
            flags
        )
        result = subprocess.run(command, cwd=current_dir, env=env)
        print(result)
    else:
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")
    

# ----- Predict -----
def run_predict(args: dict):
    # Get user input
    root         = mon.Path(args["root"])
    task         = args["task"]
    mode         = args["mode"]
    arch         = args["arch"]
    model        = args["model"]
    config       = args["config"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    torchrun     = args["torchrun"]
    master_port  = args["master_port"]
    master_addr  = args["master_addr"]
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
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model_root      = mon.parse_model_dir(arch, model)
    model           = mon.parse_model_name(model)
    data            = mon.to_list(data)
    fullname        = fullname if fullname not in [None, "None", ""] else model
    config          = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    # assert config not in [None, "None", ""]
    config  = config or ""
    weights = mon.to_str(weights, ",")
    
    for d in data:
        kwargs  = {}
        flags   = []
        kwargs |= {"--root"    : str(root)}
        kwargs |= {"--arch"    : arch}
        kwargs |= {"--model"   : model}
        kwargs |= {"--config"  : config}
        kwargs |= {"--data"    : d}
        kwargs |= {"--fullname": fullname}
        kwargs |= {"--save-dir": str(save_dir)}
        kwargs |= {"--weights" : weights}
        kwargs |= {"--device"  : device}
        # flags  += ["--torchrun"]     if torchrun     else []
        # kwargs |= {"--epochs"  : epochs}
        # kwargs |= {"--steps"   : steps}
        kwargs |= {"--seed"    : seed}
        kwargs |= {"--imgsz"   : imgsz}
        flags  += ["--resize"]       if resize       else []
        flags  += ["--benchmark"]    if benchmark    else []
        flags  += ["--save-image"]   if save_image   else []
        flags  += ["--save-debug"]   if save_debug   else []
        flags  += ["--use-fullname"] if use_fullname else []
        flags  += ["--keep-subdirs"] if keep_subdirs else []
        flags  += ["--exist-ok"]     if exist_ok     else []
        flags  += ["--verbose"]      if verbose      else []

        # Parse script file
        if use_extra_model:
            script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "i_predict.py"
            python_call = ["python"]
        else:
            script_file = current_dir / "predict.py"
            python_call = ["python"]
        
        # Parse arguments
        args_call: list[str] = []
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, list | tuple):
                args_call_ = [f"{k}={v_}" for v_ in v]
            else:
                args_call_ = [f"{k}={v}"]
            args_call += args_call_
        
        # Run prediction
        if script_file.is_py_file():
            print("\n")
            command = (
                python_call +
                [script_file] +
                args_call +
                flags
            )
            result = subprocess.run(command, cwd=current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")
        

# ----- Main -----
def main():
    defaults = vars(mon.parse_default_args(name="main"))
    menu     = menu_rich.RunmlCLI(defaults=defaults)
    args     = menu.prompt_args()
    
    # Run
    if args["mode"] in ["train"]:
        run_train(args=args)
    elif args["mode"] in ["predict"]:
        run_predict(args=args)
    else:
        raise ValueError(f"Unknown mode: {args['mode']}.")
        

if __name__ == "__main__":
    main()
