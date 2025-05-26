#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time

import torch.optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mon
import utils
from dataset_load import Dataload
from losses import *
from model import model
from spikingjelly.activation_based import functional

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark     = True

# A workaround for the bug in numpy >= 1.2.4
np.int   = np.int32
np.float = np.float64
np.bool  = np.bool_

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
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
    
    start_lr         = args["optimizer"]["lr"]
    end_lr           = args["optimizer"]["min_lr"]
    warmup_epochs    = args["optimizer"]["warmup_epochs"]
    patch_size_train = args["datamodule"]["patch_size_train"]
    patch_size_test  = args["datamodule"]["patch_size_test"]
    batch_size       = args["datamodule"]["batch_size"]
    shuffle          = args["datamodule"]["shuffle"]
    clip_grad        = args["trainer"]["clip_grad"]
    use_amp          = args["trainer"]["use_amp"]
    
    start_epoch      = 0
    optim_state_dict = None
   
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    data_root     = mon.parse_data_dir(root, data_dir=args["datamodule"]["root"])
    train_dir     = data_root / "train"
    train_dataset = Dataload(data_dir=train_dir, patch_size=patch_size_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 4,
        drop_last   = False,
        pin_memory  = True
    )
    
    if (data_root / "val").exists():
        val_dir = data_root / "val"
    elif (data_root / "test").exists():
        val_dir = data_root / "test"
    elif data in ["rain13k"]:
        val_dir = mon.ROOT_DIR / "data" / "enhance" / "rain100" / "test"
    else:
        raise ValueError("No validation dataset found.")
    val_dataset = Dataload(data_dir=val_dir, patch_size=patch_size_test)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1,
        drop_last   = False,
        pin_memory  = True
    )
    
    # Model
    model_ = model
    model_.to(device)
    if weights is not None and mon.Path(weights).is_weights_file():
        state_dict = torch.load(weights, map_location=device, weights_only=True)
        if mon.Path(weights).suffix == ".ckpt":
            state_dict       = state_dict["state_dict"]
            start_epoch      = state_dict["epoch"]
            optim_state_dict = state_dict["optimizer"]
        model_.load_state_dict(state_dict)
    functional.set_step_mode(model_, step_mode="m")
    functional.set_backend(model_,   backend="cupy")
    
    # Loss
    # criterion = nn.MSELoss().to(device)
    criterion = utils.SSIM().to(device)
    # criterion = nn.SmoothL1Loss().to(device)
    # criterion = PSNRLoss().to(device)
    
    # Optimizer
    optimizer        = optim.AdamW(model_.parameters(), lr=start_lr, eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs, eta_min=end_lr)
    scheduler        = mon.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
        
    # Training
    writer          = SummaryWriter(save_dir)
    scaler          = torch.cuda.amp.GradScaler()
    best_psnr       = 0
    best_ssim       = 0
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    iter            = 0
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss       = 0
        scaled_loss      = 0
        train_psnrs      = []
        model_.train()
        
        # Train
        with mon.create_progress_bar() as pbar:
            for i, data in pbar.track(
                sequence    = enumerate(train_loader),
                total       = len(train_loader),
                description = f"[bright_yellow] Training"
            ):
                for param in model_.parameters():
                    param.grad = None
                image    = data[0].to(device)
                ref      = data[1].to(device)
                enhanced = model_(image)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        train_ssim = criterion(enhanced, ref)
                        loss       = 1 - train_ssim
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    functional.reset_net(model_)
                else:
                    train_ssim = criterion(enhanced, ref)
                    loss       = 1 - train_ssim
                    loss.backward()
                    scaled_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    optimizer.step()
                    functional.reset_net(model_)
                torch.cuda.synchronize()
                epoch_loss += loss.item()
                iter       += 1
                for res, tar in zip(enhanced, ref):
                    train_psnrs.append(utils.torchPSNR(res, tar))
                train_psnr = torch.stack(train_psnrs).mean().item()
                train_ssim = train_ssim.item()
                
                writer.add_scalar("loss/iter_loss",  loss.item(), iter)
                writer.add_scalar("loss/epoch_loss", epoch_loss, epoch)
                writer.add_scalar("lr/epoch_loss",   scheduler.get_lr()[0], epoch)
                
            # Evaluation
            if epoch % 1 == 0:
                model_.eval()
                val_psnrs = []
                for ii, data_val in enumerate(val_loader):
                    image = data_val[0].to(device)
                    ref   = data_val[1].to(device)
                    
                    with torch.no_grad():
                        enhanced = model_(image)
                    functional.reset_net(model_)
                    
                    for res, tar in zip(enhanced, ref):
                        val_psnrs.append(utils.torchPSNR(res, tar))

                val_psnr = torch.stack(val_psnrs).mean().item()
                val_ssim = criterion(enhanced, ref).item()
                writer.add_scalar("val/psnr", val_psnr, epoch)
                writer.add_scalar("val/ssim", val_ssim, epoch)
                if val_psnr > best_psnr:
                    best_psnr       = val_psnr
                    best_psnr_epoch = epoch
                    torch.save(model_.state_dict(), str(save_dir / f"{fullname}_best_psnr.pt"))
                if val_ssim > best_ssim:
                    best_ssim       = val_ssim
                    best_ssim_epoch = epoch
                    torch.save(model_.state_dict(), str(save_dir / f"{fullname}_best_ssim.pt"))
                print("[Epoch %d Validating PSNR: %2.4f --- best_psnr_epoch %d Test_PSNR %2.4f]" % (epoch, val_psnr, best_psnr_epoch, best_psnr))
                print("[Epoch %d Validating SSIM: %2.4f --- best_ssim_epoch %d Test_SSIM %2.4f]" % (epoch, val_ssim, best_ssim_epoch, best_ssim))
            
            # Save
            torch.save(
                {
                    "epoch"     : epoch,
                    "state_dict": model_.state_dict(),
                    "optimizer" : optimizer.state_dict()
                },
                str(save_dir / f"{fullname}_last.ckpt")
            )
            torch.save(model_.state_dict(), str(save_dir / f"{fullname}_last.pt"))
            scheduler.step()
            print("-" * 150)
            print(
                "Epoch: {}\t"
                "Time: {:.4f}\t"
                "Loss: {:.4f}\t"
                "Train PSNR: {:.4f}\t"
                "Train SSIM: {:.4f}\t"
                "Learning Rate: {:.8f}\t"
                "Validate PSNR: {:.4f}\t"
                "Validate SSIM: {:.4f}".format(
                    epoch,
                    time.time() - epoch_start_time,
                    loss.item(),
                    train_psnr,
                    train_ssim,
                    scheduler.get_lr()[0],
                    val_psnr,
                    val_ssim,
                )
            )
            print("-" * 150)
    writer.close()
        

# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
