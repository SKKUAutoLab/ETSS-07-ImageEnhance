#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation utilities for models."""

__all__ = [
    "download_weights_from_url",
    "flatten_models_dict",
    "get_epoch_from_checkpoint",
    "get_global_step_from_checkpoint",
    "get_latest_checkpoint",
    "get_weights_file_from_config",
    "is_extra_model",
    "is_image",
    "list_archs",
    "list_extra_archs",
    "list_extra_models",
    "list_models",
    "list_mon_archs",
    "list_mon_models",
    "list_tasks",
    "list_weights_files",
    "load_weights",
    "parse_model_dir",
    "parse_model_fullname",
    "parse_model_name",
    "parse_weights_file",
]

from typing import Sequence
from urllib.parse import urlparse  # noqa: F401

import torch

from mon import core
from mon.constants import (
    EXTRA_MODELS, EXTRA_STR, MLType, MODELS, ROOT_DIR, Task, ZOO_DIR,
)


# ----- Retrieve (Tasks) -----
def list_tasks(project_root: str | core.Path = None) -> list[str]:
    """Lists all available tasks in the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        Sorted list of task names as strings.
    """
    tasks = Task.names()
    
    if project_root:
        default_configs = core.load_project_defaults(project_root)
        if default_configs.get("TASKS"):
            tasks = [t for t in tasks if t in default_configs["TASKS"]]
    
    return sorted(t.value for t in tasks)


# ----- Retrieve (Archs) -----
def list_mon_archs(task: str = None, mode: str = None) -> list[str]:
    """Lists all available architectures in the ``mon`` framework.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    flatten_models = flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
    
    if mode == "train":
        models = [m for m in models
                  if any(lt in MLType.trainable() for lt in flatten_models[m].mltypes)]
    
    archs = [flatten_models[m].arch.strip()
             for m in models
             if flatten_models[m].arch not in [None, "None", ""]]
    
    return sorted(core.unique(archs))


def list_extra_archs(task: str = None, mode: str = None) -> list[str]:
    """Lists all available architectures in the ``extra`` framework.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    flatten_models = flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]
    
    if mode == "train":
        models = [m for m in models
                  if any(lt in MLType.trainable() for lt in flatten_models[m]["mltypes"])]
    
    archs = [flatten_models[m]["arch"].strip()
             for m in models if flatten_models[m]["arch"] not in [None, "None", ""]]
    
    return sorted(core.unique(archs))


def list_archs(
    task: str = None,
    mode: str = None,
    project_root: str | core.Path = None
) -> list[str]:
    """Lists all available architectures in ``mon`` and ``extra`` frameworks.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    models       = list_mon_models(task=task, mode=mode)
    extra_models = list_extra_models(task=task, mode=mode)
    
    default_configs = core.load_project_defaults(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [core.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if core.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if core.snakecase(m) in project_models]
    
    flatten_mon_models   = flatten_models_dict(MODELS)
    flatten_extra_models = flatten_models_dict(EXTRA_MODELS)
    archs = (
        [flatten_mon_models[m].arch      for m in models] +
        [flatten_extra_models[m]["arch"] for m in extra_models]
    )
    archs = [a.strip() for a in archs if a not in [None, "None", ""]]
    
    return sorted(core.unique(archs))


# ----- Retrieve (Models) -----
def list_mon_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    """Lists all available models in the ``mon`` framework.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    flatten_models = flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
   
    if mode == "train":
        models = [m for m in models
                  if any(lt in MLType.trainable() for lt in flatten_models[m].mltypes)]
    
    if arch:
        models = [m for m in models if arch == flatten_models[m].arch]
        
    return sorted(models)


def list_extra_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    """Lists all available models in the ``extra`` framework.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    flatten_models = flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
   
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]

    if mode == "train":
        models = [m for m in models if any(lt in MLType.trainable() for lt in flatten_models[m]["mltypes"])]
    
    if arch:
        models = [m for m in models if arch == flatten_models[m]["arch"]]
   
    return sorted(models)


def list_models(
    task: str = None,
    mode: str = None,
    arch: str = None,
    project_root: str | core.Path = None
) -> list[str]:
    """Lists all available models in ``mon`` and ``extra`` frameworks.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    models       = list_mon_models(task=task, mode=mode, arch=arch)
    extra_models = list_extra_models(task=task, mode=mode, arch=arch)
    
    default_configs = core.load_project_defaults(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [core.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if core.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if core.snakecase(m) in project_models]
        
    for i, m in enumerate(extra_models):
        if m in models:
            extra_models[i] = f"{m} {EXTRA_STR}"
            
    return sorted(models + extra_models)


# ----- Retrieve (Checkpoint) -----
def get_latest_checkpoint(dirpath: core.Path) -> str | None:
    """Gets the latest checkpoint file path in a directory.

    Args:
        dirpath: Directory path containing checkpoint files.

    Returns:
        Path to latest checkpoint as string, or ``None`` if none found.
    """
    dirpath = core.Path(dirpath)
    ckpts   = sorted(
        (ckpt for ckpt in dirpath.files(recursive=True) if ckpt.is_torch_file()),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not ckpts:
        core.error_console.log(f"[red]Cannot find checkpoint file: {dirpath}.")
        return None
    
    return str(ckpts[0])


def get_epoch_from_checkpoint(ckpt: core.Path) -> int:
    """Gets the epoch value from a checkpoint file.

    Args:
        ckpt: Path to the checkpoint file.

    Returns:
        Epoch value from checkpoint, or ``0`` if not found or invalid.
    """
    if ckpt is None:
        return 0
    
    ckpt = core.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("epoch", 0)
    
    return 0


def get_global_step_from_checkpoint(ckpt: core.Path) -> int:
    """Gets the global step from a checkpoint file.

    Args:
        ckpt: Path to the checkpoint file.

    Returns:
        Global step from checkpoint, or ``0`` if not found or invalid.
    """
    if ckpt is None:
        return 0
    
    ckpt = core.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("global_step", 0)
    
    return 0
 
 
# ----- Retrieve (Weights) -----
def get_weights_file_from_config(config: str | core.Path) -> core.Path | None:
    """Gets the weights file path from a config file.
    
    Args:
        config: Path to the config file or a dictionary containing weights info.
    
    Returns:
        Path to the weights file as ``Path`` object.
    """
    if config is None:
        return None
    
    if not core.Path(config).is_config_file(exist=True):
        return None
        
    args    = core.load_config(config, False)
    weights = args.get("weights", None)
    return core.Path(weights) if weights else None


def list_weights_files(model: str, project_root: str | core.Path = None) -> list[core.Path]:
    """Lists weights files for a model in project and ``zoo`` dirs.

    Args:
        model: Name of model to filter weights files.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of weights file ``Path`` objects.
    """
    def collect_weights_files(root: core.Path) -> list[core.Path]:
        return sorted(f for f in root.rglob("*") if f.is_weights_file())
    
    # List all weights files in the project root and ``zoo`` directories.
    weights_files: list[core.Path] = []
    if project_root not in [None, "None", ""]:
        weights_files += collect_weights_files(core.Path(project_root) / "run" / "train")
    weights_files += collect_weights_files(ZOO_DIR)
    
    # Filter weights files by model name.
    model_name    = parse_model_name(model)
    weights_files = [f for f in weights_files if model_name in f.parts]
    
    return sorted(core.unique(weights_files))


def download_weights_from_url(url: str, path: core.Path, overwrite: bool = False) -> core.Path:
    """Downloads weights from a URL to a local path.

    Args:
        url: URL to download weights from.
        path: Local file path to save weights.
        overwrite: If ``True``, overwrites existing file. Default is ``False``.

    Returns:
        Path to downloaded weights file.

    Raises:
        ValueError: If ``url`` is not a valid URL.
    """
    if not core.Path(url).is_url():
        raise ValueError(f"[url] must be a valid URL, got {url}.")
    
    path = core.Path(path)
    if not path.exists() or overwrite:
        core.delete_files(path=path.parent, regex=path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path, None, True)
    return path


def load_weights(
    model       : torch.nn.Module,
    weights     : dict | str | core.Path,
    weights_only: bool = False
) -> dict | None:
    """Loads state dict from weights into a model.

    Args:
        model: ``torch.nn.Module`` to load weights into.
        weights: Weights as ``dict``, ``str`` path, or ``core.Path``.
        weights_only: Loads only weights if ``True``. Default is ``False``.

    Returns:
        State ``dict`` if loaded, ``None`` otherwise.
    """
    if weights is None:
        return None

    path       = core.Path(weights["path"]) if isinstance(weights, dict) and "path" in weights     else None
    state_dict = weights                    if isinstance(weights, dict) and "path" not in weights else None

    if isinstance(weights, (str, core.Path)) and core.Path(weights).is_weights_file():
        path = core.Path(weights)

    if path and path.is_weights_file():
        state_dict = torch.load(str(path), map_location=model.device, weights_only=weights_only)
    elif state_dict is None:
        core.error_console.log(f"[yellow]Cannot load weights from: {weights}[/yellow]")
        return None

    return state_dict["state_dict"] if "state_dict" in state_dict else state_dict


# ----- Convert -----
def parse_model_dir(arch: str, model: str) -> core.Path | None:
    """Parses the model's directory from given components.

    Args:
        arch: Architecture of the model.
        model: Name of the model.

    Returns:
        ``Path`` to model dir if found, else ``None``.
    """
    model_name = parse_model_name(model)
    model_dir  = (
        EXTRA_MODELS[arch][model_name].get("model_dir")
        if is_extra_model(model)
        else MODELS[arch][model_name].model_dir
    )
    return core.Path(model_dir) if model_dir else None


def parse_model_name(model: str) -> str:
    """Parses the model's name from given components.

    Args:
        model: Model name to parse.

    Returns:
        Parsed model name as a string.
    """
    return model.replace(f" {EXTRA_STR}", "").strip()


def parse_model_fullname(name: str, data: str, suffix: str = None) -> str:
    """Parses the model's full name as ``name-data-suffix`` from components.

    Args:
        name: Model's base name.
        data: Dataset name.
        suffix: Optional suffix for model name. Default is ``None``.

    Returns:
        Parsed full model name as a string.
    """
    if not name:
        core.error_console.log("[name] must be provided for the model")
    
    fullname = name
    if data:
        fullname = f"{fullname}_{data}"
    if suffix:
        _fullname = core.snakecase(fullname)
        _suffix   = core.snakecase(suffix)
        if _suffix not in _fullname:
            fullname = f"{fullname}_{core.kebabize(suffix)}"
    return fullname


def parse_weights_file(
    root   : str | core.Path,
    weights: str | core.Path | Sequence[str | core.Path]
) -> str | core.Path | Sequence[str | core.Path]:
    """Parses weights file path(s) from given components.

    Args:
        root: Root directory.
        weights: Weights file(s) to parse (str, ``Path``, or sequence).

    Returns:
        Parsed weights path(s) as single path or sequence, or ``None`` if empty.
    """
    root    = core.Path(root)
    weights = core.to_list(weights)
    
    for i, w in enumerate(weights):
        if w is not None and not core.Path(w).exists():
            weights[i] = (ROOT_DIR / w) if (ROOT_DIR / w).exists() else (root / w)
    weights = [core.Path(w) for w in weights if w not in [None, "None", ""]]

    if len(weights) == 1:
        return weights[0]
    return weights or None


def flatten_models_dict(x: dict) -> dict:
    """Flattens a nested dictionary of models.

    Args:
        x: Nested ``dict`` of models.

    Returns:
        Flattened ``dict`` with inner keys and values, adding ``arch`` key to nested
        dicts.
    """
    return {
        k2: {**v2, "arch": k1} if isinstance(v2, dict) else v2
        for k1, v1 in x.items()
        for k2, v2 in v1.items()
    }


# ----- Validity Check -----
def is_extra_model(model: str) -> bool:
    """Checks if a model is an extra model.

    Args:
        model: Name of the model to check.

    Returns:
        ``True`` if model is extra, ``False`` otherwise.
    """
    model        = model.replace(f" {EXTRA_STR}", "").strip()
    mon_models   = flatten_models_dict(MODELS)
    extra_models = flatten_models_dict(EXTRA_MODELS)
    return (
        f"{EXTRA_STR}" in model
        or (model not in mon_models and model in extra_models)
    )


def is_image(image: torch.Tensor) -> bool:
    """Checks if input is a valid image tensor.

    Args:
        image: Tensor to validate.
    
    Returns:
        ``True`` if image is valid, ``False`` otherwise.
    """
    from mon import vision
    return vision.is_image(image)
