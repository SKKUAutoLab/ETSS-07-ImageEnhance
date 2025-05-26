#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Reversible Decoupling Network for Single Image Reflection
Removal," CVPR 2025.

References:
    - https://github.com/lime-j/RDNet
"""

from models import make_model
import model as mmodel
import mon
import torch
import torch.optim
import torchvision
from options.net_options.train_options import TrainOptions

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def tensor2im(image_tensor):
    image_tensor = image_tensor.detach()
    image_numpy  = image_tensor[0].cpu().float().numpy()
    image_numpy  = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy  = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy


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
    torchrun     = args["torchrun"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    seed         = args["seed"]
    batch_size   = args["batch_size"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    save_nearby  = args["save_nearby"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    opt = TrainOptions().parse()
    opt.isTrain    = False
    opt.no_log     = True
    opt.display_id = 0
    opt.verbose    = False
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device          = mon.set_device(device)
    cudnn.benchmark = True
    opt.device      = device
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    opt.net_c_path = str(weights / "cls_model.pth")
    opt.icnn_path  = str(weights / "rdnet.pth")
    model = make_model(opt.model)
    model.initialize(opt)
    model.net_i.eval()
    model.net_c.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=dce_net)
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
            image      = datapoint["image"].to(device)
            
            # Infer
            timer.tick()
            model.set_input(
                data={
                    "input": image,
                    "fn"   : image_path,
                },
                mode="test"
            )
            output_i, output_j = model.forward()
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            enhanced = output_i
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, output_path)
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
