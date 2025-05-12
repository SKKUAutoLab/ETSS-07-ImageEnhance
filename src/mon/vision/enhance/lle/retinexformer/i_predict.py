#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Retinexformer: One-stage Retinex-based Transformer for
Low-light Image Enhancement," ICCV 2023.

References:
    - https://github.com/caiyuanhao1998/Retinexformer
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from skimage.util import img_as_ubyte

import mon
import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

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
    
    opt_path     = str(current_dir / "options" / args["opt_path"])
    opt          = parse(opt_path, is_train=False)
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    # gpu_list = ",".join(str(x) for x in args.gpus)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    device = mon.set_device(device)
    opt["dist"]   = False
    opt["device"] = device

    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model      = create_model(opt).net_g
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["params"])
    except:
        new_checkpoint = {}
        for k in checkpoint["params"]:
            new_checkpoint["module." + k] = checkpoint["params"][k]
        model.load_state_dict(new_checkpoint)
    # print("===>Testing using weights: ", weights)
    model.to(device)
    # model = nn.DataParallel(model)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=model)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer  = mon.Timer()
    factor = 4
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
            
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            if resize:
                h0, w0 = mon.image_size(image)
                image  = mon.resize(image, imgsz)
                mon.console.log("Resizing images to: ", image.shape[2], image.shape[3])
                # images = proc.resize(input=images, size=[1000, 666])
            # Padding in case images are not multiples of 4
            h, w  = mon.image_size(image)
            H, W  = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh  = H - h if h % factor != 0 else 0
            padw  = W - w if w % factor != 0 else 0
            input = F.pad(image, (0, padw, 0, padh), 'reflect')
            input = input.to(device)
            
            # Infer
            timer.tick()
            enhanced = model(input)
            timer.tock()
            
            # Post-processing
            # Unpad images to original dimensions
            enhanced = enhanced[:, :, :h, :w]
            if resize:
                enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = torch.clamp(enhanced, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                utils.save_img(str(output_path), img_as_ubyte(enhanced))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
