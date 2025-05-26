#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement," CVPR 2020.

References:
    - https://github.com/Li-Chongyi/Zero-DCE
"""

import torch
import torch.optim

import model as mmodel
import mon
import myloss

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args: dict) -> str:
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
    
    lr               = args["optimizer"]["lr"]
    weight_decay     = args["optimizer"]["weight_decay"]
    grad_clip_norm   = args["trainer"]["grad_clip_norm"]
    display_iter     = args["trainer"]["display_iter"]
    checkpoints_iter = args["trainer"]["checkpoints_iter"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)

    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    '''
    args["datamodule"] |= {
        "transform": A.Compose(transforms=[
            A.Resize(width=imgsz, height=imgsz),
        ])
    }
    '''
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(root, data_dir=args["datamodule"]["root"]),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    
    # Model
    dce_net = mmodel.enhance_net_nopool().to(device)
    dce_net.apply(weights_init)
    if weights and mon.Path(weights).is_weights_file():
        dce_net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    dce_net.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(dce_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss
    L_color = myloss.L_color()
    L_spa   = myloss.L_spa()
    L_exp   = myloss.L_exp(16, 0.6)
    L_tv    = myloss.L_TV()
    
    # Training
    with mon.create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            for i, datapoint in enumerate(train_dataloader):
                image          = datapoint["image"].to(device)
                _, enhanced, r = dce_net(image)
                
                loss_tv  = 200 * L_tv(r)
                loss_spa = torch.mean(L_spa(enhanced, image))
                loss_col =   5 * torch.mean(L_color(enhanced))
                loss_exp =  10 * torch.mean(L_exp(enhanced))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dce_net.parameters(), grad_clip_norm)
                optimizer.step()
                
                # Log
                if ((i + 1) % display_iter) == 0:
                    print("Loss at iteration", i + 1, ":", loss.item())
                
                # Save
                if ((i + 1) % checkpoints_iter) == 0:
                    torch.save(dce_net.state_dict(), save_dir / "best.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
