#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import logging

import cv2
import numpy as np
import torch
import torch.optim

import mon
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
    config_file                    = args.config_file
    # config_file                    = current_dir / "sam2_configs" / config_file
    points_per_side                = args.points_per_side               
    points_per_batch               = args.points_per_batch              
    pred_iou_thresh                = args.pred_iou_thresh               
    stability_score_thresh         = args.stability_score_thresh        
    stability_score_offset         = args.stability_score_offset        
    mask_threshold                 = args.mask_threshold                
    box_nms_thresh                 = args.box_nms_thresh                
    crop_n_layers                  = args.crop_n_layers                 
    crop_nms_thresh                = args.crop_nms_thresh               
    crop_n_points_downscale_factor = args.crop_n_points_downscale_factor
    min_mask_region_area           = args.min_mask_region_area          
    output_mode                    = args.output_mode                   
    use_m2m                        = args.use_m2m                       
    multimask_output               = args.multimask_output
    
    # Model
    sam2 = build_sam2(str(config_file), str(weights), device="cuda", apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model                          = sam2,
        points_per_side                = points_per_side,
        points_per_batch               = points_per_batch,
        pred_iou_thresh                = pred_iou_thresh,
        stability_score_thresh         = stability_score_thresh,
        stability_score_offset         = stability_score_offset,
        mask_threshold                 = mask_threshold,
        box_nms_thresh                 = box_nms_thresh,
        crop_n_layers                  = crop_n_layers,
        crop_nms_thresh                = crop_nms_thresh,
        crop_n_points_downscale_factor = crop_n_points_downscale_factor,
        min_mask_region_area           = min_mask_region_area,
        output_mode                    = output_mode,
        use_m2m                        = use_m2m,
        multimask_output               = multimask_output,
    )
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(sam2),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
      
    # Disable logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    save_dir        = save_dir / data_name
    save_dir_binary = save_dir / "binary"
    save_dir_color  = save_dir / "color"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_binary.mkdir(parents=True, exist_ok=True)
    save_dir_color.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image      = datapoint.get("input")
                meta       = datapoint.get("meta")
                image_path = meta["path"]
                timer.tick()
                masks = mask_generator.generate(image)
                timer.tock()
                # Binary
                for i, mask in enumerate(masks):
                    output_path = save_dir_binary / f"{image_path.stem}_mask_{i}.jpg"
                    cv2.imwrite(str(output_path), np.uint8(mask["segmentation"]) * 255)
                # Color
                output          = np.ones((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 4))
                output[:, :, 3] = 0
                for i, mask in enumerate(masks):
                    mask_bool         = mask["segmentation"]
                    color_mask        = np.concatenate([np.random.random(3), [1.0]])  # 0.35
                    output[mask_bool] = color_mask
                output_path = save_dir_color / f"{image_path.stem}.jpg"
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
