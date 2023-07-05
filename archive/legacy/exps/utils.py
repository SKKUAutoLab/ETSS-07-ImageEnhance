#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from shutil import copyfile
from typing import Union

from munch import Munch

from torchkit.core.fileio import create_dirs
from torchkit.core.fileio import load


# MARK: - Directories

# NOTE: Inside MLKit/exps
exps_dir        = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(exps_dir, "checkpoints")
results_dir     = os.path.join(exps_dir, "results")

# NOTE: Inside MLKit
root_dir        = os.path.dirname(exps_dir)
torchkit_dir    = os.path.join(root_dir, "torchkit")
models_zoo_dir  = os.path.join(root_dir, "models_zoo")
datasets_dir    = os.path.join(root_dir, "data")
data_dir        = os.path.join(root_dir, "data")

# NOTE: Inside workspaces
workspaces_dir = os.path.dirname(root_dir)


# MARK: - Process Config

def load_config(config: Union[str, dict]) -> Munch:
    """Load and process config from file.

    Args:
        config (str, dict):
            The config filepath that contains configuration values or the
            config dict.

    Returns:
        config (Munch):
            The config dictionary as namespace.
    """
    # NOTE: Load dictionary from file and convert to namespace using Munch
    if isinstance(config, str):
        config_dict = load(path=config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError
    
    assert config_dict is not None, f"No configuration is found at {config}!"
    config = Munch.fromDict(config_dict)
    return config


def copy_config_file(config_file: str, dst: str):
    """Copy config file to destination dir."""
    create_dirs(paths=[dst])
    copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))
