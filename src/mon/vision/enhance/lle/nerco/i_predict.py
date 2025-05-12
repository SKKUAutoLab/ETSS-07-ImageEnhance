#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Implicit Neural Representation for Cooperative Low-light
Image Enhancement," ICCV 2023.

References:
    - https://github.com/Ysz2022/NeRCo
"""

import random
from typing import Sequence

import torch.optim
from PIL import Image

import mon
from data.base_dataset import get_transform
from models import create_model
from options.test_options import TestOptions
from util import util

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
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]
    
    # Hard-code some parameters for test
    opt                = TestOptions().parse()  # get test options
    opt.num_threads    = 0       # test code only supports num_threads = 0
    opt.batch_size     = 1       # test code only supports batch_size  = 1
    opt.serial_batches = True    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip        = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id     = -1      # no visdom display; the test code saves the results to a HTML file.
    
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device     = mon.set_device(device)
    opt.device = device
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    testB_dir   = current_dir / "dataset" / "testB"
    testB_files = sorted([f for f in testB_dir.glob("*") if f.is_image_file()])
    testB_size  = len(testB_files)
    transform_A = get_transform(opt)
    transform_B = get_transform(opt)
    
    # Model
    model = create_model(opt)    # create a model given opt.model and other options
    model.setup(weights, opt)    # regular setup: load and print networks; create schedulers
    model = model.to(device)
    if opt.eval:
        model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = model.compute_efficiency_score()
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
            indexB     = random.randint(0, testB_size - 1)
            imageA     = Image.open(image_path).convert("RGB")
            imageB     = Image.open(testB_files[indexB]).convert("RGB")
            w0, h0     = imageA.size
            imageA     = transform_A(imageA).unsqueeze(0).to(device)
            imageB     = transform_B(imageB).unsqueeze(0).to(device)
            dp = {
                "A"      : imageA,
                "B"      : imageB,
                "A_paths": image_path,
                "B_paths": testB_files[indexB]
            }
            
            # Infer
            timer.tick()
            model.set_input(dp)
            model.test()
            visuals  = model.get_current_visuals()
            enhanced = visuals.get("fake_B")
            timer.tock()
            
            # Post-process
            h1, w1 = mon.image_size(enhanced)
            if h1 != h0 or w1 != w0:
                enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = util.tensor2im(enhanced)
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                Image.fromarray(enhanced).save(str(output_path))
            '''
                if save_debug:
                    if keep_subdirs:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    else:
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    output_path.mkdir(parents=True, exist_ok=True)
                    # torchvision.utils.save_image(g_a, str(output_path / f"{image_path.stem}_g_a.jpg"))
                    # torchvision.utils.save_image(pre, str(output_path / f"{image_path.stem}_pre.jpg"))
                '''
                
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
