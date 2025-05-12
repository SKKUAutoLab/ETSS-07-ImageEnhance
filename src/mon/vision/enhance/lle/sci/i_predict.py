#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
CVPR 2022.

References:
    - https://github.com/vis-opt-group/SCI
"""

import torch.utils
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.autograd import Variable

import mon
from model import Finetunemodel
from mon.nn import _size_2_t

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def compute_efficiency_score(
    model     : torch.nn.Module,
    image_size: _size_2_t = 512,
    channels  : int       = 3
) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        image_size: Input image size (H, W) or single int. Default is ``512``.
        channels: Number of input channels. Default is ``3``.

    Returns:
        Tuple of (FLOPs, parameters) as floats.
    """
    from mon import vision

    h, w   = vision.image_size(image_size)
    input  = torch.rand(1, channels, h, w).to(mon.get_model_device(model))

    flops  = FlopCountAnalysis(model, input).total()
    params = sum(p.numel() for p in model.parameters())

    return flops, params


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
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model = Finetunemodel(weights).to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = compute_efficiency_score(model=model)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
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
            image      = Variable(image).to(device)
            
            # Infer
            timer.tick()
            i, r = model(image)
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                torchvision.utils.save_image(r, str(output_path))
       
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
