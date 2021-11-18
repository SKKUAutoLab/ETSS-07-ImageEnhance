#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from typing import Union

from munch import Munch

from torchkit.core.fileio import load


# MARK: - Directories

"""Folder Hierarchy
|
|__ backup
|__ datasets    # Locally store many datasets.
|__ MLKit
|__ tools
"""

# NOTE: Inside MLKit/torchkit
torchkit_dir    = os.path.dirname(os.path.abspath(__file__))

# NOTE: Inside MLKit
root_dir        = os.path.dirname(torchkit_dir)
exps_dir        = os.path.join(root_dir, "exps")
checkpoints_dir = os.path.join(exps_dir, "checkpoints")
models_zoo_dir  = os.path.join(root_dir, "models_zoo")
datasets_dir    = os.path.join(root_dir, "data")
data_dir        = os.path.join(root_dir, "data")

# NOTE: Inside workspaces
workspaces_dir  = os.path.dirname(root_dir)


# MARK: - Process Config

def load_config(config: Union[str, dict]) -> Munch:
    """Load config as namespace.

    Args:
        config (str, dict):
            The config filepath that contains configuration values or the
            config dict.
    """
    # NOTE: Load dictionary from file and convert to namespace using Munch
    if isinstance(config, str):
        config_dict = load(path=config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError
    
    assert (config_dict is not None), f"No configuration is found at {config}!"
    config = Munch.fromDict(config_dict)
    return config
