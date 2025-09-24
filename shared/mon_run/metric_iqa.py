#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measures image quality assessment metrics for a given model and dataset."""

import argparse
import logging

import albumentations as A
import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model
import torch

import mon

mon.disable_print()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
_METRICS     = pyiqa.default_model_configs.DEFAULT_CONFIGS


# ----- PyIQA -----
def measure_metric_pyiqa(
    input_dir  : mon.Path,
    target_dir : mon.Path,
    arch       : str,
    model      : str,
    data       : str,
    device     : int | list[int] | str,
    imgsz      : int,
    resize     : bool,
    metric     : list[str],
    use_gt_mean: bool,
    verbose    : bool,
) -> dict:
    """Measure metrics using :mod:`pyiqa` package."""
    # Parse input and target directories
    input_dir  = mon.Path(input_dir)
    target_dir = mon.Path(target_dir) if target_dir else input_dir.replace("image", "ref")

    # List image files
    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = 0
    
    # Parse arguments
    device      = device[0] if len(device) == 1 else device
    device      = torch.device(("cpu" if not torch.cuda.is_available() else device))
    metric      = list(_METRICS.names()) if ("all" in metric or "*" in metric) else metric
    metric      = [m.lower() for m in metric]
    values      = {m: []     for m in metric}
    results     = {}
    h, w        = mon.image.imgsz(imgsz)
    
    # Parse metrics
    metric_f = {}
    for i, m in enumerate(metric):
        if m not in _METRICS:
            continue
        metric_f[m] = pyiqa.models.inference_model.InferenceModel(
            metric_name = m,
            as_loss     = False,
            device      = device,
        )
    
    # Prepare transform
    base_transform = A.Compose([
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    
    # Measuring
    description = f"[bright_yellow]Measuring {model} | {data} (GT Mean)" if use_gt_mean else f"[bright_yellow]Measuring {model} | {data}"
    with mon.create_progress_bar(transient=not verbose) as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = description
        ):
            # Image
            image  = mon.image.load_image(path=image_file)
            h0, w0 = mon.image.imgsz(image)
            h2, w2 = h, w
            
            # Target
            target      = None
            target_file = None
            need_resize = resize
            for ext in mon.ImageExtension.values():
                temp = target_dir / f"{image_file.stem}{ext}"
                if temp.exists():
                    target_file = temp
            if target_file and target_file.exists():  # Has target file
                target = mon.image.load_image(path=target_file)
                h1, w1 = mon.image.imgsz(target)
                if h1 != h0 or w1 != w0:  # Mismatch size between image and target
                    h2, w2      = h1, w1
                    need_resize = True
            
            # Transform
            transform = base_transform
            if need_resize:
                transform = A.Resize(height=h2, width=w2) + transform
            if target is not None:
                transform.add_targets(additional_targets={"target": "image"})
                augmented = transform(image=image, target=target)
                image     = augmented["image"]
                target    = augmented["target"]
            else:
                augmented = transform(image=image)
                image     = augmented["image"]
            
            # Move to device
            image  =  image.unsqueeze(0).to(device=device)
            target = target.unsqueeze(0).to(device=device) if target is not None else None
            
            # Measure metric
            for m in metric:
                if m not in _METRICS:
                    continue
                if target is None and _METRICS[m]["metric_mode"] == "FR":
                    continue
                elif target is not None and _METRICS[m]["metric_mode"] == "FR":
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))

            num_items += 1

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / num_items)
        else:
            results[m] = None
    return results


def update_best_results(results: dict, new_values: dict) -> dict:
    for m, v in new_values.items():
        if m in _METRICS:
            lower_better = _METRICS[m].get("lower_better", False)
            if m not in results:
                results[m] = v
            elif results[m] is None:
                results[m] = v
            elif v:
                results[m] = min(results[m], v) if lower_better else max(results[m], v)
    return results


# ----- Main -----
# @click.command(name="metric", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# @click.option("--input-dir",   type=click.mon.Path(exists=True),  default=None, help="Image directory.")
# @click.option("--target-dir",  type=click.mon.Path(exists=False), default=None, help="Ground-truth directory.")
# @click.option("--result-file", type=str,                      default=None, help="Result file.")
# @click.option("--arch",        type=str,                      default=None, help="Model's architecture.")
# @click.option("--model",       type=str,                      default=None, help="Model's fullname.")
# @click.option("--data",        type=str,                      default=None, help="Source data.")
# @click.option("--device",      type=str,                      default=None, help="Running devices.")
# @click.option("--imgsz",       type=int,                      default=512)
# @click.option("--resize",      is_flag=True)
# @click.option("--metric",      type=str, multiple=True, help="Measuring metric.")
# @click.option("--use-gt-mean", is_flag=True)
# @click.option("--backend",     type=click.Choice(["pyiqa"], case_sensitive=False), default=["pyiqa"], multiple=True)
# @click.option("--save-txt",    is_flag=True)
# @click.option("--verbose",     is_flag=True)
def main(
    input_dir  : mon.Path,
    target_dir : mon.Path,
    result_file: mon.Path,
    arch       : str,
    model      : str,
    data       : str,
    device     : int | list[int] | str,
    imgsz      : int,
    resize     : bool,
    metric     : list[str],
    use_gt_mean: bool,
    backend    : list[str],
    save_txt   : bool,
    verbose    : bool,
):
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    mon.console.rule(f"[bold red] {model}")
    mon.console.log(f"[bold green]Model : {model}")
    mon.console.log(f"[bold red]Data  : {data}")
    mon.console.log(f"[bold]Device: {device}")

    input_dir  = mon.Path(input_dir)  if input_dir  else None
    target_dir = mon.Path(target_dir) if target_dir else None

    if not input_dir or not input_dir.is_dir():
        raise ValueError(f"``input_dir`` does not exist: {input_dir}.")

    results         = {}
    results_gt_mean = {}
    backend         = [backend] if not isinstance(backend, list) else backend
    backend         = [str(b).lower() for b in backend]

    for b in backend:
        if b in ["pyiqa"]:
            metric_values = measure_metric_pyiqa(
                input_dir   = input_dir,
                target_dir  = target_dir,
                arch        = arch,
                model       = model,
                data        = data,
                device      = device,
                imgsz       = imgsz,
                resize      = resize,
                metric      = metric,
                use_gt_mean = False,
                verbose     = verbose,
            )
            results = update_best_results(results, metric_values)
            if use_gt_mean:
                metric_values_gt_mean = measure_metric_pyiqa(
                    input_dir   = input_dir,
                    target_dir  = target_dir,
                    arch        = arch,
                    model       = model,
                    data        = data,
                    device      = device,
                    imgsz       = imgsz,
                    resize      = resize,
                    metric      = metric,
                    use_gt_mean = True,
                    verbose     = verbose,
                )
                results_gt_mean = update_best_results(results_gt_mean, metric_values_gt_mean)
        else:
            mon.console.log(f"`{backend}` is not supported!")
    
    # Show results
    message = ""
    # Headers
    for m, v in results.items():
        if v:
            message += f"{f'{m}':<10}\t"
    message += "\n"
    # Values
    for i, (m, v) in enumerate(results.items()):
        if v:
            if i == len(results) - 1:
                message += f"{v:.10f}\n"
            else:
                message += f"{v:.10f}\t"
    for i, (m, v) in enumerate(results_gt_mean.items()):
        if v:
            if i == len(results) - 1:
                message += f"{v:.10f}\n"
            else:
                message += f"{v:.10f}\t"
    print(f"{message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric")
    parser.add_argument("--input-dir",   type=str, help="Input image directory.")
    parser.add_argument("--target-dir",  type=str, help="Ground-truth image directory.")
    parser.add_argument("--result-file", type=str, help="Result file.")
    parser.add_argument("--arch",        type=str, help="Model's architecture.")
    parser.add_argument("--model",       type=str, help="Model's fullname.")
    parser.add_argument("--data",        type=str, help="Source data name.")
    parser.add_argument("--device",      type=str, help="Running devices.")
    parser.add_argument("--imgsz",       type=int, default=512)
    parser.add_argument("--resize",      action="store_true")
    parser.add_argument("--metric",      type=str, action="append", help="Measuring metric.")
    parser.add_argument("--use-gt-mean", action="store_true")
    parser.add_argument("--backend",     choices=["pyiqa"], default="pyiqa")
    parser.add_argument("--save-txt",    action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    main(**vars(args))
