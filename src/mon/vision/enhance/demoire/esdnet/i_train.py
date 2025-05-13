#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    https://github.com/CVMI-Lab/UHDM
"""

import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

import mon
from dataset.load_data import *
from model.model import model_fn_decorator
from model.nets import my_model
from utils.common import *
from utils.loss_util import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train_epoch(args, train_img_loader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    """Training Loop for each epoch"""
    tbar       = tqdm(train_img_loader)
    lr         = optimizer.state_dict()["param_groups"][0]["lr"]
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for batch_idx, data in enumerate(tbar):
        loss, psnr, ssim = model_fn(args, data, model, iters)
        # Backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters          += 1
        total_loss     += loss.item()
        total_psnr     += psnr
        total_ssim     += ssim
        avg_train_loss  = total_loss / (batch_idx + 1)
        avg_train_psnr  = total_psnr / (batch_idx + 1)
        avg_train_ssim  = total_ssim / (batch_idx + 1)
        desc            = ("Training: Epoch %d, lr %.7f, Avg. Loss = %.5f, Avg. PSNR = %.5f, Avg. SSIM = %.5f"
                           % (epoch, lr, avg_train_loss, avg_train_psnr, avg_train_ssim))
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    # the learning rate is adjusted after each epoch
    lr_scheduler.step()
    return lr, avg_train_loss, avg_train_psnr, avg_train_ssim, iters


def load_checkpoint(model, optimizer, load_epoch):
    state_dict = torch.load(load_epoch)
    mon.console.log("Loading pre-trained checkpoint %s" % load_epoch)
    model_state_dict = state_dict["state_dict"]
    optimizer_dict   = state_dict["optimizer"]
    learning_rate    = state_dict["learning_rate"]
    iters            = state_dict["iters"]
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_dict)
    mon.console.log("Learning rate recorded from the checkpoint: %s" % str(learning_rate))
    return learning_rate, iters


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
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args["GENERAL"]["GPU_ID"]
    
    # Seed
    random.seed(args["GENERAL"]["SEED"])
    np.random.seed(args["GENERAL"]["SEED"])
    torch.manual_seed(args["GENERAL"]["SEED"])
    torch.cuda.manual_seed_all(args["GENERAL"]["SEED"])
    if args["GENERAL"]["SEED"] == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
    
    # Data I/O
    args["DATA"]["TRAIN_DATASET"] = str(mon.ROOT_DIR / args["DATA"]["TRAIN_DATASET"])
    args["DATA"]["TEST_DATASET"]  = str(mon.ROOT_DIR / args["DATA"]["TEST_DATASET"])
    train_path       = args["DATA"]["TRAIN_DATASET"]
    train_img_loader = create_dataset(args, data_path=train_path, mode="train")
    
    # Model
    if weights not in [None, ""]:
        weights = mon.Path(weights)
        if not weights.is_ckpt_file(exist=True):
            if (root / weights).is_ckpt_file(exist=True):
                weights = root / weights
            if (root / "run" / "train" / weights).is_ckpt_file(exist=True):
                weights = root / "run" / "train" / weights
    mon.console.log(weights)
    
    model = my_model(
        en_feature_num = args["MODEL"]["EN_FEATURE_NUM"],
        en_inter_num   = args["MODEL"]["EN_INTER_NUM"],
        de_feature_num = args["MODEL"]["DE_FEATURE_NUM"],
        de_inter_num   = args["MODEL"]["DE_INTER_NUM"],
        sam_number     = args["MODEL"]["SAM_NUMBER"],
    ).to(device)
    model._initialize_weights()
    
    # Optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "initial_lr": args["SOLVER"]["BASE_LR"]}],
    )
    learning_rate = args["SOLVER"]["BASE_LR"]
    iters = 0
    if weights is not None:
        learning_rate, iters = load_checkpoint(model, optimizer, weights)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0        = args["SOLVER"]["T_0"],
        T_mult     = args["SOLVER"]["T_MULT"],
        eta_min    = args["SOLVER"]["ETA_MIN"],
        last_epoch = args["TRAIN"]["LOAD_EPOCH"] - 1
    )
    
    # Loss
    loss_fn  = multi_VGGPerceptualLoss(lam=args["TRAIN"]["LAM"], lam_p=args["TRAIN"]["LAM_P"]).to(device)
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device)
    
    # Logger
    logger = SummaryWriter(str(save_dir))
    
    # Training
    best_loss = 100.0
    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(args["TRAIN"]["LOAD_EPOCH"] + 1, args["SOLVER"]["EPOCHS"] + 1):
        learning_rate, avg_train_loss, avg_train_psnr, avg_train_ssim, iters = (
            train_epoch(args, train_img_loader, model, model_fn, optimizer, epoch, iters, lr_scheduler)
        )
        logger.add_scalar("train/avg_loss",      avg_train_loss, epoch)
        logger.add_scalar("train/learning_rate", learning_rate,  epoch)
        
        # Save the best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
        if avg_train_psnr > best_psnr:
            best_psnr = avg_train_psnr
            torch.save(model.state_dict(), save_dir / "best_psnr.pt")
        if avg_train_ssim > best_ssim:
            best_ssim = avg_train_ssim
            torch.save(model.state_dict(), save_dir / "best_ssim.pt")
        
        # Save the latest model
        torch.save({
            "learning_rate": learning_rate,
            "iters"        : iters,
            "optimizer"    : optimizer.state_dict(),
            "state_dict"   : model.state_dict()
        }, save_dir / "last.ckpt")
        torch.save(model.state_dict(), save_dir / "last.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
