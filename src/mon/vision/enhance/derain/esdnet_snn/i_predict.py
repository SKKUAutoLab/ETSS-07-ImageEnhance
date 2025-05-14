#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
import torch.optim
import torchvision
from torch.nn import functional as F

import model as M
import mon
from spikingjelly.activation_based import functional

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def get_score_map(b, c, h, w, is_mean: bool = True) -> torch.Tensor:
    center_h = h / 2
    center_w = w / 2
    score    = torch.ones((b, c, h, w))
    if not is_mean:
        for h in range(h):
            for w in range(w):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def split_image(
    img_tensor  : torch.Tensor,
    crop_size   : int = 80,
    overlap_size: int = 8
) -> tuple[list, list]:
    b, c, h, w = img_tensor.shape
    
    h_starts = [x for x in range(0, h, crop_size - overlap_size)]
    while h_starts[-1] + crop_size >= h:
        h_starts.pop()
    h_starts.append(h - crop_size)
    
    w_starts = [x for x in range(0, w, crop_size - overlap_size)]
    while w_starts[-1] + crop_size >= w:
        w_starts.pop()
    w_starts.append(w - crop_size)
   
    starts     = []
    split_data = []
    for hs in h_starts:
        for ws in w_starts:
            c_img_data = img_tensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(c_img_data)
    return split_data, starts


def merge_image(split_data, starts, crop_size, shape=(1, 3, 80, 80)) -> torch.Tensor:
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    tot_score  = torch.zeros((b, c, h, w))
    merge_img  = torch.zeros((b, c, h, w))
    score_map  = get_score_map(b, c, crop_size, crop_size, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += score_map * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += score_map
    merge_img = merge_img / tot_score
    return merge_img


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

    crop_size    = imgsz[0]  # 80
    overlap_size = 8         # 8
    pad_size     = 16        # 16 * 2 = 32
    
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
    model = M.model.to(device)
    functional.set_step_mode(model, step_mode="m")
    functional.set_backend(model,   backend="cupy")
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    if mon.Path(weights).suffix == ".ckpt":
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=model)
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
            h0, w0     = mon.image_size(image)
            if resize:
                image = mon.resize(image, imgsz)
            image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode="constant", value=0)
            b, c, h1, w1 = image.shape
            
            # Infer
            timer.tick()
            split_data, starts = split_image(image, crop_size=crop_size, overlap_size=overlap_size)
            for j, data in enumerate(split_data):
                split_data[j] = model(data).to(device)
                split_data[j] = split_data[j].cpu()
                functional.reset_net(model)
            enhanced = merge_image(split_data, starts, crop_size=crop_size, shape=(b, c, h1, w1))
            enhanced = torch.clamp(enhanced, 0, 1)
            timer.tock()
            
            # Post-processing
            enhanced = enhanced[:, :, pad_size:-pad_size, pad_size:-pad_size]
            if h1 != h0 or w1 != w0:
                enhanced = mon.resize(enhanced, (h0, w0))
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(enhanced, str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
