#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements utilities for data I/O classes and functions."""

__all__ = [
    "parse_data_loader",
    "list_mon_datasets",
    "list_extra_datasets",
    "list_datasets",
]

from mon import core, vision
from mon.constants import DATASETS, EXTRA_DATASETS, Split, Task, ROOT_DIR


# ----- Retrieve -----
def list_mon_datasets(task: str, mode: str) -> list[str]:
    """Lists all available datasets in the ``mon`` framework.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = DATASETS

    return sorted([
        d for d in datasets
        if task in datasets[d].tasks and split in datasets[d].splits
    ])


def list_extra_datasets(task: str, mode: str) -> list[str]:
    """Lists all available datasets in the ``extra`` framework.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = EXTRA_DATASETS

    return sorted([
        d for d in datasets
        if task in datasets[d]["tasks"] and split in datasets[d]["splits"]
    ])


def list_datasets(
    task: str,
    mode: str,
    project_root: str | core.Path = None
) -> list[str]:
    """Lists all available datasets.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).
        project_root: Root directory of the project. Default is ``None``.

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    datasets        = sorted(list_mon_datasets(task, mode) + list_extra_datasets(task, mode))
    default_configs = core.load_project_defaults(project_root)
    if default_configs.get("DATASETS"):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets


# ----- Convert -----
def parse_data_loader(
    src      : core.Path | str,
    data_root: core.Path | str = None,
    to_tensor: bool = False,
    verbose  : bool = False
) -> tuple[str, core.Dataset]:
    """Parses I/O worker for data src.

    Args:
        src: Source of input data.
        data_root: Dataset root dir (e.g., ``data/ntire_2025_llie``). Default is ``None``.
        to_tensor: If ``True``, converts to tensor. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``False``.

    Returns:
        Tuple of data name, loader, and writer.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    src = core.Path(src)

    if src.stem in DATASETS:
        src  = src.stem
        root = core.parse_data_dir(root=data_root, data_dir=src)
        # if root and not root.is_dir():
        #     root = root / "data"
        config = {
            "name"     : src,
            "root"     : root,
            "split"    : Split.TEST,
            "to_tensor": to_tensor,
            "verbose"  : verbose,
        }
        data_name   = src
        data_loader = DATASETS.build(config=config)
    elif src.is_dir():
        data_name   = src.name
        data_loader = vision.ImageLoader(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
    elif src.is_video_file():
        data_name   = src.name
        data_loader = vision.VideoLoaderCV(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
    else:
        raise ValueError(f"[src] is invalid: {src}.")
    
    return data_name, data_loader
