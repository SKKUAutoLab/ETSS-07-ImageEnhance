#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "UNet++ Based Nested Skip Connections Network for Low-Light
Image Enhancement,"

References:
    - https://github.com/xiwang-online/LLUnetPlusPlus
"""

from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mon
from average_meter import AverageMeter
from loss import Loss
from model import NestedUNet

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train_epoch(train_dataloader, model, criterion, optimizer, device):
    loss_meters = AverageMeter()
    model.train()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(train_dataloader),
            total       = len(train_dataloader),
            description = f"[bright_yellow] Training"
        ):
            input  = datapoint["image"].to(device)
            target = datapoint["ref_image"].to(device)
            meta   = datapoint["meta"]
            output = model(input)
            loss   = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meters.update(loss.item(), input.size(0))
            # mon.console.log(loss_meters.avg)
    return loss_meters.avg


def val_epoch(val_dataloader, model, criterion, device):
    loss_meters = AverageMeter()
    psnr_meters = mon.PeakSignalNoiseRatio().to(device)
    ssim_meters = mon.StructuralSimilarityIndexMeasure().to(device)
    model.eval()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(val_dataloader),
            total       = len(val_dataloader),
            description = f"[bright_yellow] Validating"
        ):
            input  = datapoint["image"].to(device)
            target = datapoint["ref_image"].to(device)
            meta   = datapoint["meta"]
            output = model(input)
            loss   = criterion(output, target)
            loss_meters.update(loss.item(), input.size(0))
            psnr_meters.update(output, target)
            ssim_meters.update(output, target)
            # mon.console.log(loss_meters.avg)
    return loss_meters.avg, psnr_meters.compute(), ssim_meters.compute()


def train(args: dict) -> str:
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
    
    lr           = args["optimizer"]["lr"]
    weight_decay = args["optimizer"]["weight_decay"]
    loss_weights = args["loss"]["loss_weights"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    cudnn.benchmark = True
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader
    
    # Model
    model = NestedUNet().to(device)
    model.train()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    # Loss
    criterion = Loss(*loss_weights).to(device)
    
    # Logging
    writer = SummaryWriter(log_dir=str(save_dir))
    log    = OrderedDict([
        ("epoch"     , []),
        ("lr"        , []),
        ("train/loss", []),
        ("val/loss"  , []),
        ("val/psnr"  , []),
        ("val/ssim"  , []),
    ])
    best_loss = 1000
    best_psnr = 0
    best_ssim = 0
    
    # Training
    for epoch in range(epochs):
        train_loss  = train_epoch(train_dataloader, model, criterion, optimizer, device)
        val_results = val_epoch(val_dataloader,     model, criterion, device)
        val_loss    = float(val_results[0])
        val_psnr    = float(val_results[1].cpu().detach().numpy())
        val_ssim    = float(val_results[2].cpu().detach().numpy())
        scheduler.step()
        mon.console.log(
            "Epoch [%d/%d] train/loss %.4f - val/loss %.4f - val/psnr %.4f - val/ssim %.4f\n"
            % (epoch, epochs, train_loss, val_loss, val_psnr, val_ssim)
        )
        
        # Log
        log["epoch"].append(epoch)
        log["lr"].append(lr)
        log["train/loss"].append(train_loss)
        log["val/loss"].append(val_loss)
        log["val/psnr"].append(val_psnr)
        log["val/ssim"].append(val_ssim)
        pd.DataFrame(log).to_csv(str(save_dir / "log.csv"))
        writer.add_scalars(
            "train",
            {"train/loss": train_loss},
            epoch,
        )
        writer.add_scalars(
            "val",
            {
                "val/loss": val_loss,
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
            },
            epoch,
        )
        
        # Save
        if val_loss < best_loss:
            torch.save(model.state_dict(), str(save_dir / "best.pt"))
            best_loss = val_loss
        if val_psnr > best_psnr:
            torch.save(model.state_dict(), str(save_dir / "best_psnr.pt"))
            best_psnr = val_psnr
        if val_ssim > best_ssim:
            torch.save(model.state_dict(), str(save_dir / "best_ssim.pt"))
            best_ssim = val_ssim
        torch.save(model.state_dict(), str(save_dir / "last.pt"))
        torch.cuda.empty_cache()
   
    writer.close()
    
    
# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
