#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements main running pipeline."""

import subprocess
from typing import Collection, Sequence

import click

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
modes 	     = ["train", "predict"]


# ----- Utils -----
def parse_menu_string(items: Sequence | Collection, num_columns: int = 4) -> str:
    """Parses a list of items into a formatted menu string.

    Args:
        items: Items to display in the menu.
        num_columns: Number of columns for menu layout. Default is ``4``.

    Returns:
        Formatted menu string.
    """
    s = "\n  "
    for i, item in enumerate(items):
        s += f"{f'{i}.':>6} {item}\n  "
    s += f"{f'Other.':} (please specify)\n  "
    return s


# ----- Train -----
def run_train(args: dict):
    # Get user input
    task         = args["task"]
    mode         = args["mode"]
    config       = args["config"]
    root         = mon.Path(args["root"])
    arch         = args["arch"]
    model        = args["model"]
    # data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    # imgsz        = args["imgsz"]
    # resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    # use_fullname = args["use_fullname"]
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
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    assert config not in [None, "None", ""]
    # save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    weights  = mon.to_str(weights, ",")
    
    kwargs   = {
        "--config"  : config,
        "--root"    : str(root),
        "--arch"    : arch,
        "--model"   : model,
        "--fullname": fullname,
        "--save-dir": str(save_dir),
        "--weights" : weights,
        "--device"  : device,
        # "--imgsz"   : imgsz,
        "--epochs"  : epochs,
        "--steps"   : steps,
    }
    flags  = ["--benchmark"]    if benchmark    else []
    flags += ["--save-image"]   if save_image   else []
    flags += ["--save-debug"]   if save_debug   else []
    flags += ["--keep-subdirs"] if keep_subdirs else []
    flags += ["--exist-ok"]     if exist_ok     else []
    flags += ["--verbose"]      if verbose      else []
    
    # Parse script file
    if use_extra_model:
        # torch_distributed_launch = mon.EXTRA_MODELS[arch][model]["torch_distributed_launch"]
        script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "i_train.py"
        python_call = ["python"]
        # device      = mon.parse_device(device)
        # if isinstance(device, list) and torch_distributed_launch:
        #     python_call = [
        #         f"python",
        #         f"-m",
        #         f"torch.distributed.launch",
        #         f"--nproc_per_node={str(len(device))}",
        #         f"--master_port=9527"
        #     ]
    else:
        script_file = current_dir / "train.py"
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
    
    # Run training
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
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")
    

# ----- Predict -----
def run_predict(args: dict):
    # Get user input
    task         = args["task"]
    mode         = args["mode"]
    config       = args["config"]
    root         = mon.Path(args["root"])
    arch         = args["arch"]
    model        = args["model"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    # epochs       = args["epochs"]
    # steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    # use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model_root      = mon.parse_model_dir(arch, model)
    model           = mon.parse_model_name(model)
    fullname        = fullname if fullname not in [None, "None", ""] else model
    config          = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    # assert config not in [None, "None", ""]
    config   = config or ""
    weights  = mon.to_str(weights, ",")
    
    for d in data:
        kwargs  = {
            "--config"  : config,
            "--root"    : str(root),
            "--arch"    : arch,
            "--model"   : model,
            "--data"    : d,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--weights" : weights,
            "--device"  : device,
            "--imgsz"   : imgsz,
            # "--epochs"  : epochs,
            # "--steps"   : steps,
        }
        flags   = ["--resize"]       if resize       else []
        flags  += ["--benchmark"]    if benchmark    else []
        flags  += ["--save-image"]   if save_image   else []
        flags  += ["--save-debug"]   if save_debug   else []
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
@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",         type=click.Path(exists=True),    help="Project root.")
@click.option("--task",         type=str,     default=None,      help="Task.")
@click.option("--mode",         type=str,     default="predict", help="Mode: train | predict.")
@click.option("--arch",         type=str,     default=None,      help="Model architecture or family.")
@click.option("--model",        type=str,     default=None,      help="Model name.")
@click.option("--config",       type=str,     default=None,   	 help="Config file.")
@click.option("--data",         type=str,     default=None,      help="Dataset name or directory.")
@click.option("--fullname",     type=str,     default=None,   	 help="Full name of the current run.")
@click.option("--save-dir",     type=str,     default=None,      help="Optional saving directory.")
@click.option("--weights",      type=str,     default=None,      help="Path to the pretrained weights.")
@click.option("--device",       type=str,     default=None,      help="Running device (i.e., cuda:0 | cuda:0,1,2,3 | cpu).")
@click.option("--imgsz",        type=int,     default=-1,        help="Image size for the model.")
@click.option("--resize",       is_flag=True,                    help="Resize the input image to `imgsz`.")
@click.option("--epochs",       type=int,     default=-1,   	 help="Training epochs.")
@click.option("--steps",        type=int,     default=-1,   	 help="Training steps.")
@click.option("--benchmark",    is_flag=True,                    help="Benchmark the model.")
@click.option("--save-image",   is_flag=True,                    help="Save the output image.")
@click.option("--save-debug",   is_flag=True,                    help="Save the debug information.")
@click.option("--use-fullname", is_flag=True,                    help="Use the full name for the save_dir.")
@click.option("--keep-subdirs", is_flag=True,                    help="Keep subdirectories in the save_dir.")
@click.option("--exist-ok",     is_flag=True,                    help="Exist OK.")
@click.option("--verbose",      is_flag=True,                    help="Verbosity.")
def main(
    root        : str,
    task        : str,
    mode        : str,
    arch        : str,
    model       : str,
    config      : str,
    data        : str,
    fullname    : str,
    save_dir    : str,
    weights     : str,
    device      : int | list[int] | str,
    imgsz       : int,
    resize      : bool,
    epochs      : int,
    steps       : int,
    benchmark   : bool,
    save_image  : bool,
    save_debug  : bool,
    use_fullname: bool,
    keep_subdirs: bool,
    exist_ok    : bool,
    verbose     : bool,
):
    # Start
    click.echo(click.style(f"\nInput Prompt:", fg="white", bg="red", bold=True))
    # Task
    tasks_       = mon.list_tasks(project_root=root)
    tasks_str_   = parse_menu_string(tasks_)
    task         = click.prompt(click.style(f"Task {tasks_str_}", fg="bright_green", bold=True), default=task)
    task         = tasks_[int(task)] if mon.is_int(task) else task
    # Mode
    mode         = click.prompt(click.style(f"Mode {parse_menu_string(modes)}", fg="bright_green", bold=True), default=mode)
    mode         = modes[int(mode)] if mon.is_int(mode) else mode
    # Architecture
    archs_       = mon.list_archs(project_root=root, task=task, mode=mode)
    archs_str_   = parse_menu_string(archs_)
    arch	     = click.prompt(click.style(f"Architecture {archs_str_}", fg="bright_green", bold=True), type=str, default=arch)
    arch 	     = archs_[int(arch)] if mon.is_int(arch) else arch
    # Model
    models_      = mon.list_models(project_root=root, task=task, mode=mode, arch=arch)
    models_str_  = parse_menu_string(models_)
    model	     = click.prompt(click.style(f"Model {models_str_}", fg="bright_green", bold=True), type=str, default=model)
    model 	     = models_[int(model)] if mon.is_int(model) else model
    model_name   = mon.parse_model_name(model)
    # Config
    model_dir    = mon.parse_model_dir(arch, model)
    configs_     = mon.list_configs(project_root=root, model_root=model_dir, model=model, absolute_path=True)
    configs_str_ = parse_menu_string(configs_)
    config	     = click.prompt(click.style(f"Config {configs_str_}", fg="bright_green", bold=True), type=str, default="")
    config       = configs_[int(config)] if mon.is_int(config) else config
    config_args  = mon.load_config(config, False)
    # Weights
    weights_     = mon.list_weights_files(project_root=root, model=model)
    weights_str_ = parse_menu_string(weights_)
    weights      = weights or config_args.get("weights", "")
    weights      = click.prompt(click.style(f"Weights {weights_str_}", fg="bright_green", bold=True), type=str, default=weights)
    weights      = weights if weights not in [None, ""] else None
    if weights:
        if isinstance(weights, str):
            weights = mon.to_list(weights)
        weights  = [weights_[int(w)] if mon.is_int(w) else w for w in weights]
        weights  = [w.replace("'", "") for w in weights]
    # Data (predict)
    if mode in ["predict"]:
        data_     = mon.list_datasets(project_root=root, task=task, mode="predict")
        data_str_ = parse_menu_string(data_)
        data      = data.replace(",", ",\n    ") if isinstance(data, str) else data
        data	  = click.prompt(click.style(f"Predict(s) {data_str_}", fg="bright_green", bold=True), type=str, default=data)
        data 	  = mon.to_list(data)
        data 	  = [data_[int(d)] if mon.is_int(d) else d for d in data]
    # Fullname
    fullname    = fullname or (mon.Path(config).stem if config not in [None, "None", ""] else model_name)
    fullname    = click.prompt(click.style(f"Fullname: {fullname}", fg="bright_green", bold=True), type=str, default=fullname)
    # Device
    devices_    = mon.list_devices()
    devices_str = parse_menu_string(devices_)
    device      = "auto" if model_name in mon.list_mon_models(mode=mode, task=task) and mode == "train" else device
    device      = click.prompt(click.style(f"Device {devices_str}", fg="bright_green", bold=True), type=str, default=device or "cuda:0")
    device	    = devices_[int(device)] if mon.is_int(device) else device
    # Predict Flags
    if mode in ["predict"]:  # Image size
        # imgsz  = imgsz or config_args.get("imgsz", -1)
        imgsz  = click.prompt(click.style(f"Image size         ", fg="bright_yellow", bold=True), type=str, default=imgsz)
        imgsz  = mon.to_int_list(imgsz)
        imgsz  = imgsz[0] if len(imgsz) == 1 else imgsz
        imgsz  = None if imgsz < 0 else imgsz
        resize = "y" if resize else "n"
        resize = click.prompt(click.style(f"Resize?       [y/n]", fg="bright_yellow", bold=True), type=str, default=resize)
        resize = True if resize == "y" else False
    # Training Flags
    if mode in ["train"]:  # Epochs
        epochs = click.prompt(click.style(f"Epochs             ", fg="bright_yellow", bold=True), type=int, default=epochs)
        epochs = None if epochs < 0 else epochs
        steps  = click.prompt(click.style(f"Steps              ", fg="bright_yellow", bold=True), type=int, default=steps)
        steps  = None if steps  < 0 else steps
    benchmark    = "y" if benchmark else "n"
    benchmark    = click.prompt(click.style(f"Benchmark?    [y/n]", fg="bright_yellow", bold=True), type=str, default=benchmark)
    benchmark    = True if benchmark == "y" else False
    save_image   = "y" if save_image else "n"
    save_image   = click.prompt(click.style(f"Save image?   [y/n]", fg="bright_yellow", bold=True), type=str, default=save_image)
    save_image   = True if save_image == "y" else False
    save_debug   = "y" if save_debug else "n"
    save_debug   = click.prompt(click.style(f"Save debug?   [y/n]", fg="bright_yellow", bold=True), type=str, default=save_debug)
    save_debug   = True if save_debug == "y" else False
    keep_subdirs = "y" if keep_subdirs else "n"
    keep_subdirs = click.prompt(click.style(f"Use fullpath? [y/n]", fg="bright_yellow", bold=True), type=str, default=keep_subdirs)
    keep_subdirs = True if keep_subdirs == "y" else False
    # Common Flags
    # Exist OK?
    exist_ok = "y" if exist_ok else "n"
    exist_ok = click.prompt(click.style(f"Exist OK?     [y/n]", fg="bright_yellow", bold=True), type=str, default=exist_ok)
    exist_ok = True if exist_ok == "y" else False
    # Use Verbose
    verbose  = "y" if verbose else "n"
    verbose  = click.prompt(click.style(f"Verbosity?    [y/n]", fg="bright_yellow", bold=True), type=str, default=verbose)
    verbose  = True if verbose  == "y" else False
    
    # Run
    if mode in ["train"]:
        args = {
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "root"        : root,
            "arch"        : arch,
            "model"       : model,
            # "data"        : None,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "weights"     : weights,
            "device"      : device,
            # "imgsz"       : imgsz,
            # "resize"      : resize,
            "epochs"      : epochs,
            "steps"       : steps,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "save_debug"  : save_debug,
            "use_fullname": use_fullname,
            "keep_subdirs": keep_subdirs,
            "exist_ok"    : exist_ok,
            "verbose"     : verbose,
        }
        run_train(args=args)
    elif mode in ["predict"]:
        args = {
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "root"        : root,
            "arch"        : arch,
            "model"       : model,
            "data"        : data,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "weights"     : weights,
            "device"      : device,
            "imgsz"       : imgsz,
            "resize" 	  : resize,
            # "epochs"      : epochs,
            # "steps"       : steps,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "save_debug"  : save_debug,
            "use_fullname": use_fullname,
            "keep_subdirs": keep_subdirs,
            "exist_ok"    : exist_ok,
            "verbose"     : verbose,
        }
        run_predict(args=args)
    else:
        raise ValueError(f":param:`mode` must be one of {modes}, but got {mode}.")
        

if __name__ == "__main__":
    main()

# endregion
