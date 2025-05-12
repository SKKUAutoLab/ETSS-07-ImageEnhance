#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/arsenyinfo/EnlightenGAN-inference
# pip install onnx-tool
# https://pypi.org/project/onnx-tool/0.1.7/

import cv2
import torch

import mon
from onnx_model import EnlightenOnnxModel

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, False, verbose=False)
    
    # Model
    model = EnlightenOnnxModel(weights=weights)
    
    # Measure efficiency score
    # if benchmark:
    #    model  = EnlightenOnnxModel(weights=weights)
    #    inputs = {"input": create_ndarray_f32((1, 3, 512, 512)), }
    #    onnx_tool.model_profile(str(current_dir/"enlighten_inference/enlighten.onnx"), inputs, None)
    
    # Predicting
    timer = mon.Timer()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            image      = datapoint["image"]
            
            # Infer
            timer.tick()
            enhanced = model.predict(image)
            timer.tock()
            
            # Post-processing
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                cv2.imwrite(str(output_path), enhanced)
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
