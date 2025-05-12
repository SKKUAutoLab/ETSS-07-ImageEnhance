#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Interpretable Optimization-Inspired Unfolding Network for
Low-light Image Enhancement," IEEE TPAMI 2025.

References:
    - https://github.com/AndersonYong/URetinex-Net-PLUS
"""

import argparse

import torch
import torch.nn as nn

import mon
from network.Math_Module import P, Q
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


def get_params(decom_low, r, l, adjust, fusion):
    param  = 0
    param += param_self_compute(decom_low)
    param += param_self_compute(r)
    param += param_self_compute(l)
    param += param_self_compute(adjust)
    if fusion is not None:
        param += param_self_compute(fusion)
    return param


class Inference(nn.Module):
    
    def __init__(self, opts):
        super().__init__()
        self.opts = argparse.Namespace(**opts)
        # Loading R; old_model_opts; and L model
        self.unfolding_model_opts, self.model_R, self.model_L = load_unfolding(self.opts)
        # Loading adjustment model, fusion_model
        self.fusion_opts, self.adjust_model, self.fusion_model = load_AdjustFusion(self.opts)
        # Loading decomposition model
        self.model_Decom_low, self.model_Decom_high = load_decom(self.fusion_opts)
        
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        
        print("total parameters: ", get_params(self.model_Decom_low, self.model_R, self.model_L, self.adjust_model, self.fusion_model))
    
    def get_ratio(self, high_l, low_l):
        bs, c, w, h = low_l.shape
        ratio_maps = []
        for i in range(bs):
            ratio_mean = (high_l[i, :, :, :] / (low_l[i, :, :, :]+0.0001)).mean()
            if "min_ratio" in self.fusion_opts:
                ratio_mean = max(ratio_mean, self.fusion_opts.min_ratio)
                assert ratio_mean >= self.fusion_opts.min_ratio
            ratio_maps.append(torch.ones((1, c, w, h)).cuda() * ratio_mean)
        return torch.cat(ratio_maps, dim=0)
    
    def make_high_L(self, input_high_img):
        if self.model_Decom_high is not None:
            [R_high, Q_high] = self.model_Decom_high(input_high_img)
        else:
            Q_high, _ =  torch.max(input_high_img, dim=1)
            Q_high    = Q_high.unsqueeze(0)
            R_high    = input_high_img / (Q_high + 0.0001)
        return R_high, Q_high
    
    def ratio_maker(self, L, input_high_img):
        if input_high_img is None:
            ratio = torch.ones(L.shape).cuda() * self.opts.ratio
        else:
            _, Q_high = self.make_high_L(input_high_img)
            ratio     = self.get_ratio(high_l=Q_high, low_l=L)
        return ratio
    
    def unfolding(self, input_low_img, return_component=False):
        P_results, Q_results, R_results, L_results = [[], [], [], []]
        for t in range(self.unfolding_model_opts.round):
            if t == 0:  # init P, Q
                P, Q = self.model_Decom_low(input_low_img)
            else:  # update P, Q
                w_p  = (self.unfolding_model_opts.gamma + self.unfolding_model_opts.Roffset * t)
                w_q  = (self.unfolding_model_opts.lamda + self.unfolding_model_opts.Loffset * t)
                P    = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q    = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            # update R, L
            # update R, L
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
            P_results.append(P)
            Q_results.append(Q)
            R_results.append(R)
            L_results.append(L)
        if not return_component:
            return R_results, L_results
        else:
            return P_results, Q_results, R_results, L_results
    
    def unfolding_inference(self, input_low_img):
        R_results, L_results = self.unfolding(input_low_img, return_component=False)
        results = []
        for t in range(len(R_results)):
            if (t+1) in self.fusion_opts.fusion_layers:
                # print("fusion layer : %d"%(t+1))
                results.append(R_results[t])
        assert len(results) == len(self.fusion_opts.fusion_layers)
        return results, L_results[-1]
    
    def adjust_and_fusion(self, inter_R_list, L, ratio):
        High_L    = self.adjust_model(l=L, alpha=ratio)
        R_enhance = None
        if self.fusion_model is not None:
            I_enhance, R_enhance = self.fusion_model(inter_R_list, High_L)
        else:
            assert len(inter_R_list) == 1
            I_enhance = inter_R_list[-1] * High_L
        return High_L, I_enhance, R_enhance
    
    def forward(self, input_low_img, input_high_img=None):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
            if input_high_img is not None:
                input_high_img = input_high_img.cuda()
        with torch.no_grad():
            start = time.time()
            R_results, L = self.unfolding_inference(input_low_img=input_low_img)
            ratio  = self.ratio_maker(L, input_high_img)
            High_L, I_enhance, R_enhance = self.adjust_and_fusion(inter_R_list=R_results, L=L, ratio=ratio)
            p_time = (time.time() - start)
        return I_enhance, p_time
    
    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
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
    
    args["Decom_model_high_path"]         = mon.ZOO_DIR / args["decom_model_high_weights"]
    args["Decom_model_low_path"]          = mon.ZOO_DIR / args["decom_model_low_weights"]
    args["fusion_model_A_path"]           = mon.ZOO_DIR / args["fusion_weights"]
    args["pretrain_unfolding_model_path"] = mon.ZOO_DIR / args["pretrain_unfolding_weights"]
    
    '''
    state_dict = torch.load(str(args["fusion_model_A_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["fusion_model_A_path"]))
    
    state_dict = torch.load(str(args["pretrain_unfolding_model_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["pretrain_unfolding_model_path"]))
    '''
    
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
