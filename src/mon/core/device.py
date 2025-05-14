#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles device management and memory usage."""

__all__ = [
    "get_cuda_memory_usages",
    "get_memory_usages",
    "get_model_device",
    "is_rank_zero",
    "list_devices",
    "parse_device",
    "pynvml_available",
    "set_device",
]

import os
from typing import Any

import psutil
import torch

from mon.constants import MemoryUnit
from mon.core.rich import console
from mon.core.type_extensions import generate_combinations

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

CUDA_PREFIX = "cuda:"


# ----- Retrieve -----
def list_devices() -> list[str]:
    """Lists all available devices on the machine.

    Returns:
        List of device strings including ``auto``, ``cpu``, and CUDA if available.
    """
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices <= 0:
            return devices
        # Add all CUDA device combinations (e.g., ``cuda:0``, ``cuda:1``, ``cuda:0,1``, etc.)
        cuda_indices      = list(range(num_devices))
        cuda_combinations = generate_combinations(cuda_indices)
        devices.extend([f"{CUDA_PREFIX}{','.join(str(i) for i in comb)}" for comb in cuda_combinations])
    return devices


def get_cuda_memory_usages(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets GPU memory status as a list of total, used, and free memory.

    Args:
        device: GPU device index. Default is ``0``.
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.name_to_byte()[unit]
    return [
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    ]


def get_memory_usages(unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets RAM status as a list of total, used, and free memory.

    Args:
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.name_to_byte()[MemoryUnit.from_value(unit)]
    return [
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    ]


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Gets the device of a model's parameters.

    Args:
        model: Model to check.

    Returns:
        ``torch.device`` where model parameters reside.
    """
    return next(model.parameters()).device


# ----- Update -----
def set_device(device: Any) -> torch.device | str:
    """Sets the device for the current process.

    Args:
        device: Device to set (e.g., CUDA index, list, or string).

    Returns:
        Selected ``torch.device``, defaults to ``cpu`` if CUDA unavailable.
    """
    if isinstance(device, torch.device):
        return device

    device = parse_device(device)

    if device in ["auto", "cuda"]:
        return device
    if device == "cpu":
        return torch.device("cpu")
    if isinstance(device, list):
        console.log(f"Using the first available device in {device}: {device[0]}")
        device = device[0]
    return torch.device(f"cuda:{device[0]}")


# ----- Convert -----
def parse_device(device: Any) -> torch.device | str | list[str]:
    """Parses a device spec into a list or string.

    Args:
        device: Device to parse (e.g., ``torch.device``, int, str, or ``None``).

    Returns:
        torch.device.
        str: ``auto``, ``cpu``, or ``cuda``.
        list[str]: List of cuda device indices (e.g., ``['0', '1']``).
    """
    if isinstance(device, torch.device):
        return device
    if device in [None, "", "cpu"]:
        return "cpu"
    if device in ["auto", "cuda"]:
        return device

    if isinstance(device, int):
        device = [str(device)]
    if isinstance(device, str):
        device = (device.lower()
                  .replace("cuda:", "")
                  .translate(str.maketrans("", "", "()[ ]' ")))
        device = device.split(",")
        device = [str(i) for i in device]

    return device


# ----- Validation Check -----
def is_rank_zero() -> bool:
    """Checks if current process is rank zero in distributed training.

    Notes:
        Based on PyTorch Lightning's DDP documentation, "LOCAL_RANK" and "NODE_RANK"
        environment variables indicate child processes for GPUs. Absence of both
        denotes the main process (rank zero).

    Returns:
        ``True`` if current process is rank zero, ``False`` otherwise.
    """
    return "LOCAL_RANK" not in os.environ and "NODE_RANK" not in os.environ
