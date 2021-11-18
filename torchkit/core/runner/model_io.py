#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model Loading and Writing.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from torchkit.core.fileio import get_latest_file
from torchkit.core.fileio import is_torch_saved_file
from torchkit.core.fileio import is_url
from torchkit.utils import models_zoo_dir

logger = logging.getLogger()


# MARK: - State Dict

def load_state_dict_from_path(
    path  		: Optional[str],
    model_dir   : Optional[str] = models_zoo_dir,
    map_location: Optional[str] = torch.device("cpu"),
    progress	: bool 		    = True,
    check_hash	: bool			= False,
    file_name 	: Optional[str] = None,
    **_
) -> Optional[Union[dict, OrderedDict]]:
    """Load state dict at the given URL. If downloaded file is a zip file, it
    will be automatically decompressed. If the object is already present in
    `model_dir`, it's deserialized and returned. The default value of
    `model_dir` is `MLKit/models_zoo`.

    Args:
        path (str, optional):
            URL of the object to download.
        model_dir (string, optional):
            Directory in which to save the object.
        map_location (optional):
            A function or a dict specifying how to remap storage locations
            (see torch.load)
        progress (bool, optional):
            Whether or not to display a progress bar to stderr.
            Default: `True`.
        check_hash (bool, optional):
            If `True`, the filename part of the URL should follow the naming
            convention `filename-<sha256>.ext` where ``<sha256>`` is the
            first eight or more digits of the SHA256 hash of the contents of
            the file. The hash is used to ensure unique names and to verify
            the contents of the file. Default: `False`.
        file_name (str, optional):
            Name for the downloaded file. Filename from `url` will be used
            if not set.

    Example:
        >>> state_dict = load_state_dict_from_path(
        >>> 	'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
        >>> )
    """
    if path is None:
        return None
    
    state_dict = None
    if is_torch_saved_file(path=path):
        # Can be either the weight file or the weights file.
        state_dict = torch.load(path, map_location=map_location)
    elif is_url(path=path):
        state_dict = load_state_dict_from_url(
            url=path, model_dir=model_dir, map_location=map_location,
            progress=progress, check_hash=check_hash, file_name=file_name
        )
    
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if "state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    return state_dict


def intersect_state_dict(
    da	   : OrderedDict[str, torch.Tensor],
    db	   : OrderedDict[str, torch.Tensor],
    exclude: Union[tuple, list] = ()
) -> OrderedDict[str, torch.Tensor]:
    """Dictionary intersection of matching keys and shapes, omitting 'exclude'
    keys, using da values.
    """
    return {
        k: v for k, v in da.items()
        if k in db and
           not any(x in k for x in exclude) and
           v.shape == db[k].shape
    }


def match_state_dict(
    model_dict	   : OrderedDict[str, torch.Tensor],
    pretrained_dict: OrderedDict[str, torch.Tensor],
    exclude		   : Union[tuple, list] = ()
) -> OrderedDict[str, torch.Tensor]:
    """Filter out unmatched keys btw the model's `state_dict` and the
    pretrained's `state_dict`. Omitting `exclude` keys.

    Args:
        model_dict (OrderedDict):
            The model's `state_dict`.
        pretrained_dict (OrderedDict)"
            The pretrained's `state_dict`.
        exclude (tuple, list):
            List of excluded keys. Default: `()`.
            
    Returns:
        model_dict (OrderedDict):
            The filtered model's `state_dict`.
    """
    # 1. Filter out unnecessary keys
    intersect_dict = intersect_state_dict(pretrained_dict, model_dict, exclude)
    """
       intersect_dict = {
           k: v for k, v in pretrained_dict.items()
           if k in model_dict and
              not any(x in k for x in exclude) and
              v.shape == model_dict[k].shape
       }
       """
    # 2. Overwrite entries in the existing state dict
    model_dict.update(intersect_dict)
    return model_dict


def load_state_dict(
    module	  : nn.Module,
    state_dict: OrderedDict[str, torch.Tensor],
    strict    : bool = False,
    **_
) -> nn.Module:
    """Load the module state dict. This is an extension from
    `nn.Module.load_state_dict()`. We add an extra snippet to drop missing keys
    between module's state_dict and pretrained state_dict, which will cause
    an error.

    Args:
        module (nn.Module):
            The module to load state dict.
        state_dict (dict):
            A dict containing parameters and  persistent buffers.
        strict (bool):
            Whether to strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: `False`.

    Returns:
        module (nn.Module):
            The module after loading state dict.
    """
    module_dict = module.state_dict()
    module_dict = match_state_dict(model_dict=module_dict,
                                   pretrained_dict=state_dict)
    module.load_state_dict(module_dict, strict=strict)
    return module


# MARK: - Load Pretrained

def load_pretrained(
    module	  	: nn.Module,
    path  		: Optional[str],
    model_dir   : Optional[str] = models_zoo_dir,
    map_location: Optional[str] = None,
    progress	: bool 		    = True,
    check_hash	: bool			= False,
    file_name	: Optional[str] = None,
    strict		: bool			= False,
    **_
) -> nn.Module:
    """Load pretrained weights. This is a very convenient function to load the
    state dict from pretrained weights file. Filter out mismatch keys and then
    load the layers' weights.
    
    Args:
        module (nn.Module):
            The module to load pretrained.
        path (str, optional):
            URL of the object to download.
        model_dir (string, optional):
            Directory in which to save the object.
        map_location (optional):
            A function or a dict specifying how to remap storage locations (
            see torch.load)
        progress (bool, optional):
            Whether or not to display a progress bar to stderr. Default: `True`.
        check_hash (bool, optional):
            If `True`, the filename part of the URL should follow the naming
            convention `filename-<sha256>.ext` where `<sha256>` is the first
            eight or more digits of the SHA256 hash of the contents of the
            file. The hash is used to ensure unique names and to verify the
            contents of the file. Default: `False`.
        file_name (str, optional):
            Name for the downloaded file. Filename from `url` will be used
            if not set.
        strict (bool):
            Whether to strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: `False`.
    """
    state_dict = load_state_dict_from_path(
        path, model_dir, map_location, progress, check_hash, file_name
    )
    module = load_state_dict(module, state_dict, strict=strict)
    return module


load_checkpoint = load_pretrained
load_weights	= load_pretrained


# MARK: - Checkpoint

def get_latest_checkpoint(
    dirpath: Union[str, Path], name: str = "*.ckpt"
) -> Optional[str]:
    """Get the latest weights in the `dirpath`.

    Args:
        dirpath (str, Path):
            The dirpath that contains the checkpoints.
        name (str):
            The name or pattern of the weights file.

    Returns:
        ckpt (str, optional):
            The weights filepath.
    """
    if ".ckpt" not in name:
        name += ".ckpt"
    ckpt = get_latest_file(path=os.path.join(dirpath, name))
    if ckpt is None:
        logger.warning(f"Cannot find weights file.")
    return ckpt


def get_epoch(ckpt: Union[str, Path]) -> int:
    """Get the current epoch from the saved weights file.

    Args:
        ckpt (str, Path):
            The weights filepath.

    Returns:
        epoch (int):
            The current epoch.
    """
    if isinstance(ckpt, Path):
        ckpt = str(ckpt)
    
    epoch = 0
    if is_torch_saved_file(path=ckpt):
        ckpt = torch.load(ckpt)
        epoch = getattr(ckpt, "epoch", 0)
    
    return epoch


def get_global_step(ckpt: Union[str, Path]) -> int:
    """Get the global step from the saved weights file.

    Args:
        ckpt (str, Path):
            The weights filepath.

    Returns:
        global_step (int):
            The global step.
    """
    if isinstance(ckpt, Path):
        ckpt = str(ckpt)
    
    global_step = 0
    if is_torch_saved_file(path=ckpt):
        ckpt = torch.load(ckpt)
        global_step = getattr(ckpt, "global_step", 0)
    
    return global_step


# MARK: - Weights

def ckpt_to_weights(
    ckpt: str, model_dir: Optional[str] = None, file_name: Optional[str] = None
):
    """Convert the `.ckpt` to a `.pth` file.
    
    Args:
        ckpt (str):
            The checkpoint file.
        model_dir (str, optional):
            The dir to save the weights file.
        file_name (str, optional):
            The name of the weight file.
    """
    state_dict = load_state_dict_from_path(ckpt)
    if model_dir is None:
        model_dir = str(Path(ckpt).parent)
    if file_name is None:
        file_name = str(Path(ckpt).name)
    file_name = f"{str(Path(ckpt).stem)}.pth"
    save_path = os.path.join(model_dir, file_name)
    torch.save(state_dict, save_path)
