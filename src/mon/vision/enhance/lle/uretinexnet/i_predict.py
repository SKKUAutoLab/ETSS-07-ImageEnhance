#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "URetinex-Net: Retinex-based Deep Unfolding Network for
Low-light-Image-Enhancement," CVPR 2022.

References:
    - https://github.com/AndersonYong/URetinex-Net
"""

import time

import torchvision.transforms as transforms

import mon
from mon import nn
from network.decom import Decom
from network.Math_Module import P, Q
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


class Inference(nn.Module):
    
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # Loading decomposition model
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts["decom_model_low_weights"])
        # Loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts["unfolding_model_weights"])
        # Loading adjustment model
        self.adjust_model    = load_adjustment(self.opts["adjust_model_weights"])
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
            # transforms.Resize(1280),
        ]
        self.transform = transforms.Compose(transform)
        # mon.console.log(self.model_Decom_low)
        # mon.console.log(self.model_R)
        # mon.console.log(self.model_L)
        # mon.console.log(self.adjust_model)
        # time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0:  # Initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else:  # Update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P   = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q   = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start_time = time.time()
            R, L       = self.unfolding(input_low_img)
            High_L     = self.illumination_adjust(L, self.opts["ratio"])
            I_enhance  = High_L * R
            run_time   = (time.time() - start_time)
        return I_enhance, run_time

    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
        """
        file_name = os.path.basename(self.opts.img_path)
        name      = file_name.split('.')[0]
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_%d_URetinexNet"%(name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)
        mon.console.log("================================= time for %s: %f============================"%(file_name, p_time))
        """
        return enhance, run_time
        

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
    
    args["decom_model_low_weights"] = mon.ZOO_DIR / args["decom_model_low_weights"]
    args["unfolding_model_weights"] = mon.ZOO_DIR / args["unfolding_model_weights"]
    args["adjust_model_weights"]    = mon.ZOO_DIR / args["adjust_model_weights"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model = Inference(args).to(device)
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
            image_path = meta["path"]
            
            # Infer
            timer.tick()
            enhanced, _ = model.run(image_path)
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                torchvision.utils.save_image(enhanced, str(output_path))
        
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
