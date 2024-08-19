#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import cv2
import numpy as np
import torch
import torch.optim

import mon
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data                           = args.data
    save_dir                       = args.save_dir
    weights                        = args.weights
    device                         = mon.set_device(args.device)
    imgsz                          = args.imgsz[0]
    resize                         = args.resize
    benchmark                      = False  # args.benchmark
    save_image                     = args.save_image
    save_debug                     = args.save_debug
    use_fullpath                   = args.use_fullpath
    model_type                     = "vit_t"
    points_per_side                = args.points_per_side
    points_per_batch               = args.points_per_batch
    pred_iou_thresh                = args.pred_iou_thresh
    stability_score_thresh         = args.stability_score_thresh
    stability_score_offset         = args.stability_score_offset
    box_nms_thresh                 = args.box_nms_thresh
    crop_n_layers                  = args.crop_n_layers
    crop_nms_thresh                = args.crop_nms_thresh
    crop_n_points_downscale_factor = args.crop_n_points_downscale_factor
    min_mask_region_area           = args.min_mask_region_area
    output_mode                    = args.output_mode

    # Model
    sam = sam_model_registry[model_type](checkpoint=weights)
    sam = sam.to(device).eval()
    mask_generator = SamAutomaticMaskGenerator(
        model                          = sam,
        points_per_side                = points_per_side,
        points_per_batch               = points_per_batch,
        pred_iou_thresh                = pred_iou_thresh,
        stability_score_thresh         = stability_score_thresh,
        stability_score_offset         = stability_score_offset,
        box_nms_thresh                 = box_nms_thresh,
        crop_n_layers                  = crop_n_layers,
        crop_nms_thresh                = crop_nms_thresh,
        crop_n_points_downscale_factor = crop_n_points_downscale_factor,
        min_mask_region_area           = min_mask_region_area,
        output_mode                    = output_mode,
    )
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(sam),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint.get("image")
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                
                # Infer
                timer.tick()
                masks = mask_generator.generate(image)
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        relative_path   = image_path.relative_path(data_name)
                        binary_save_dir = save_dir / relative_path.parent / "binary"
                        color_save_dir  = save_dir / relative_path.parent / "color"
                    else:
                        binary_save_dir = save_dir / data_name / "binary"
                        color_save_dir  = save_dir / data_name / "color"
                    # Binary
                    for i, mask in enumerate(masks):
                        output_path = binary_save_dir / f"{image_path.stem}_mask_{i}.jpg"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path), np.uint8(mask["segmentation"]) * 255)
                    # Color
                    output          = np.ones((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 4))
                    output[:, :, 3] = 0
                    for i, mask in enumerate(masks):
                        mask_bool         = mask["segmentation"]
                        color_mask        = np.concatenate([np.random.random(3), [1.0]])  # 0.35
                        output[mask_bool] = color_mask
                    output_path = color_save_dir / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), np.uint8(output * 255))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
