#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parses config arguments and command line arguments."""

__all__ = [
    "parse_default_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket

from mon.core import pathlib
from mon.core.config import utils
from mon.core.config.core import CLI_OPTIONS
from mon.core.device import parse_device


# ----- Parse Args -----
def parse_default_args(name: str = "main") -> argparse.Namespace:
    """Parse default arguments."""
    parser = argparse.ArgumentParser(description=name)
    
    for opt_name, opt_params in CLI_OPTIONS.items():
        action      = opt_params.get("action",      "store")
        default     = opt_params.get("default",     None)
        opt_type    = opt_params.get("type",        None)
        choices     = opt_params.get("choices",     None)
        required    = opt_params.get("required",    False)
        help_text   = opt_params.get("help",        "")
        prompt_only = opt_params.get("prompt_only", False)  # Use in interactive CLI only, not parse_args
        
        if prompt_only:
            continue
        '''
        if opt_type == bool and default is None:
            default = False
        if action == "store_true" and default is None:
            default = False
        if action == "store_false" and default is None:
            default = True
        '''
        
        kwargs = {
            "action"  : action,
            "default" : default,
            "required": required,
            "help"    : help_text,
        }
        if action in ["store_true", "store_false"]:
            kwargs.pop("default")
        if opt_type:
            kwargs["type"] = opt_type
        if choices:
            kwargs["choices"] = choices
        flag = f"--{opt_name.replace("_", "-")}"
        parser.add_argument(flag, **kwargs)
    
    '''
    parser.add_argument("--config",       type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--root",         type=_str_or_none, default=None, help="Root directory of the current run.")
    parser.add_argument("--arch",         type=_str_or_none, default=None, help="Model architecture.")
    parser.add_argument("--model",        type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--data",         type=_str_or_none, default=None, help="Dataset name or directory.")
    parser.add_argument("--fullname",     type=_str_or_none, default=None, help="Full name of the current run.")
    parser.add_argument("--save-dir",     type=_str_or_none, default=None, help="Saving directory. If not set, it will be determined.")
    parser.add_argument("--weights",      type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",       type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--imgsz",        type=_int_or_none, default=None, help="Image sizes.")
    parser.add_argument("--resize",       action="store_true",             help="Resize the input image to `imgsz`.")
    parser.add_argument("--epochs",       type=_int_or_none, default=None, help="Training epochs.")
    parser.add_argument("--steps",        type=_int_or_none, default=None, help="Training steps.")
    parser.add_argument("--benchmark",    action="store_true",             help="Benchmark the model.")
    parser.add_argument("--save-image",   action="store_true",             help="Save the output image.")
    parser.add_argument("--save-debug",   action="store_true",             help="Save the debug information.")
    parser.add_argument("--use-fullname", action="store_true",             help="Use the full name for the save_dir.")
    parser.add_argument("--keep-subdirs", action="store_true",             help="Keep subdirectories in the save_dir.")
    parser.add_argument("--exist-ok",     action="store_true",             help="If ``False``, it will delete the save directory if it already exists.")
    parser.add_argument("--verbose",      action="store_true",             help="Verbose mode.")
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER,        help="Additional arguments")
    '''
    
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return parser.parse_args()


def parse_train_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for training."""
    from mon import nn
    
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_default_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = utils.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = utils.load_config(config)
    
    # Prioritize cli_args -> args
    root         = root                     or args["root"]
    arch         = cli_args["arch"]         or args["arch"]
    model        = cli_args["model"]        or args["model"]
    data         = cli_args["data"]         or args["data"]
    fullname     = cli_args["fullname"]     or args["fullname"]
    save_dir     = cli_args["save_dir"]     or args["save_dir"]
    weights      = cli_args["weights"]      or args["weights"]
    device       = cli_args["device"]       or args["device"]
    imgsz        = cli_args["imgsz"]        or args["imgsz"]
    resize       = cli_args["resize"]       or args["resize"]
    epochs       = cli_args["epochs"]       or args["epochs"]
    steps        = cli_args["steps"]        or args["steps"]
    benchmark    = cli_args["benchmark"]    or args["benchmark"]
    save_image   = cli_args["save_image"]   or args["save_image"]
    save_debug   = cli_args["save_debug"]   or args["save_debug"]
    use_fullname =                             args["use_fullname"]
    keep_subdirs = cli_args["keep_subdirs"] or args["keep_subdirs"]
    exist_ok     = cli_args["exist_ok"]     or args["exist_ok"]
    verbose      = cli_args["verbose"]      or args["verbose"]
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        if use_fullname:
            save_dir = pathlib.parse_save_dir(root/"run"/"train", arch, model, fullname)
        else:
            save_dir = pathlib.parse_save_dir(root/"run"/"train", arch, model, data)
    else:
        save_dir = pathlib.Path(save_dir)
        if str("run/train") not in str(save_dir):
            save_dir = pathlib.Path(f"run/train/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
            
    # weights = nn.parse_weights_file(root/"run"/"train", weights)
    weights = nn.parse_weights_file(root, weights)
    device  = parse_device(device)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["keep_subdirs"] = keep_subdirs
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args


def parse_predict_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for predicting."""
    from mon import vision, nn
    
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_default_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = utils.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = utils.load_config(config)
    
    # Prioritize cli_args -> args
    root         = root                     or args["root"]
    arch         = cli_args["arch"]         or args["arch"]
    model        = cli_args["model"]        or args["model"]
    data         = cli_args["data"]         or args["data"]
    fullname     = cli_args["fullname"]     or args["fullname"]
    save_dir     = cli_args["save_dir"]     or args["save_dir"]
    weights      = cli_args["weights"]      or args["weights"]
    device       = cli_args["device"]       or args["device"]
    imgsz        = cli_args["imgsz"]        or args["imgsz"]
    resize       = cli_args["resize"]       or args["resize"]
    epochs       = cli_args["epochs"]       or args["epochs"]
    steps        = cli_args["steps"]        or args["steps"]
    benchmark    = cli_args["benchmark"]    or args["benchmark"]
    save_image   = cli_args["save_image"]   or args["save_image"]
    save_debug   = cli_args["save_debug"]   or args["save_debug"]
    use_fullname =                             args["use_fullname"]
    keep_subdirs = cli_args["keep_subdirs"] or args["keep_subdirs"]
    exist_ok     = cli_args["exist_ok"]     or args["exist_ok"]
    verbose      = cli_args["verbose"]      or args["verbose"]
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        if use_fullname:
            save_dir = pathlib.parse_save_dir(root/"run"/"predict", arch, fullname, None)
        else:
            save_dir = pathlib.parse_save_dir(root/"run"/"predict", arch, model,    None)
    else:
        save_dir = pathlib.Path(save_dir)
        save_dir = save_dir.replace("run/train/", "")
        if str("run/predict") not in str(save_dir):
            save_dir = pathlib.Path(f"run/predict/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
        
    weights = nn.parse_weights_file(root, weights)
    device  = parse_device(device)
    imgsz   = vision.image_size(imgsz)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["keep_subdirs"] = keep_subdirs
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args
