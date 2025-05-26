#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "D-FINE: Redefine Regression Task of DETRs as
Fine-grained Distribution Refinement," ICLR 2025.

References:
    - https://github.com/Peterande/D-FINE
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model         = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


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

    conf_thres   = args["conf_thres"]

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
    resume = mon.parse_weights_file(root, args["resume"]) if args["resume"] else None
    if weights and weights.is_weights_file(exist=True):
        resume = weights

    cfg_path     = current_dir / "options" / args["cfg_path"]
    update_dict  = {"resume": str(resume)} if resume else {}
    update_dict |= {
        "device"      : device,
        "seed"        : seed,
        "__include__" : args.get("__include__", None),
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(root), **update_dict)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(device)

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
            size0      = torch.tensor([[w0, h0]]).to(device)
            input      = mon.resize(image, imgsz)

            # Infer
            timer.tick()
            labels, boxes, scores = model(input, size0)
            timer.tock()

            # Post-process
            labels = [l.cpu().numpy() for l in labels]
            boxes  = [b.cpu().numpy() for b in  boxes]
            scores = [s.cpu().numpy() for s in scores]

            # Save Result
            if save_result:
                output_dir = mon.parse_output_dir(save_dir, data_name, mon.SAVE_LABEL_DIR, image_path, keep_subdirs, save_nearby)
                label_path = output_dir / f"{image_path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(label_path), "w") as f:
                    for j, img in enumerate(image):
                        ss = scores[j]
                        cs = labels[j][ss >= conf_thres]
                        bs =  boxes[j][ss >= conf_thres]
                        if len(bs) == 0:
                            continue
                        bs = mon.convert_bbox(bbox=bs, code=mon.ShapeCode.VOC2YOLO, height=h0, width=w0)
                        for c, b, s in zip(cs, bs, ss):
                            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")

            # Save Image
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, mon.SAVE_VISUALIZE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                # mon.save_image(enhanced, str(output_path))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
