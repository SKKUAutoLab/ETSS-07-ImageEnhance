#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import socket
import time
from copy import deepcopy
from threading import Thread

import click
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mon
import my_test as test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import (
    check_file, check_img_size, colorstr, fitness, fitness_ap,
    fitness_ap50, fitness_f1, fitness_p, fitness_r, get_latest_run, increment_path,
    init_seeds, labels_to_class_weights,
    labels_to_image_weights, one_cycle, print_mutation, set_logging, strip_optimizer,
)
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_evolution, plot_images, plot_results
from utils.torch_utils import intersect_dicts, is_parallel, ModelEMA, select_device, torch_distributed_zero_first
from utils.wandb_logging.wandb_utils import check_wandb_resume, WandbLogger

logger        = logging.getLogger(__name__)
console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    weights          = opt.weights
    weights          = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir         = mon.Path(opt.save_dir)
    epochs           = opt.epochs
    batch_size       = opt.batch_size
    total_batch_size = opt.total_batch_size
    rank             = opt.global_rank
    freeze           = opt.freeze
    
    # Directories
    wdir         = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last         = wdir     / "last.pt"
    best         = wdir     / "best.pt"
    best_p       = wdir     / "best_p.pt"
    best_r       = wdir     / "best_r.pt"
    best_f1      = wdir     / "best_f1.pt"
    best_ap50    = wdir     / "best_ap50.pt"
    best_ap      = wdir     / "best_ap.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # Create plots
    cuda  = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data, encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        _train    = data_dict["train"]
        _val      = data_dict["val"]
        _test     = data_dict["test"]
        if isinstance(_train, list):
            _train = [str(mon.DATA_DIR / t) for t in _train]
        elif _train:
            _train = str(mon.DATA_DIR / _train)
        if isinstance(_val, list):
            _val   = [str(mon.DATA_DIR / t) for t in _val]
        elif _val:
            _val   = str(mon.DATA_DIR / _val)
        if isinstance(_test, list):
            _test  = [str(mon.DATA_DIR / t) for t in _test]
        elif _test:
            _test  = str(mon.DATA_DIR / _test)
        data_dict["train"] = _train
        data_dict["val"]   = _val
        data_dict["test"]  = _test
    is_coco = opt.data.endswith("coco.yaml")

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {"wandb": None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp          = hyp  # add hyperparameters
        run_id           = torch.load(weights, map_location=device).get("wandb_id") if weights.endswith(".pt") and os.path.isfile(weights) else None
        wandb_logger     = WandbLogger(opt, mon.Path(opt.save_dir).stem, run_id, data_dict)
        loggers["wandb"] = wandb_logger.wandb
        data_dict        = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
    
    nc    = 1 if opt.single_cls else int(data_dict["nc"])  # number of classes
    names = ["item"] if opt.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt       = torch.load(weights, map_location=device)  # load checkpoint
        model      = Model(opt.model or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude    = ["anchor"] if (opt.model or hyp.get("anchors")) and not opt.resume else []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info("Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.model, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    # with torch_distributed_zero_first(rank):
        # check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path  = data_dict["val"]

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, "im"):
            if hasattr(v.im, "implicit"):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, "imc"):
            if hasattr(v.imc, "implicit"):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, "imb"):
            if hasattr(v.imb, "implicit"):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, "imo"):
            if hasattr(v.imo, "implicit"):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, "ia"):
            if hasattr(v.ia, "implicit"):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, "attn"):
            if hasattr(v.attn, "logit_scale"):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, "q_bias"):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, "v_bias"):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, "relative_position_bias_table"):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, "rbr_dense"):
            if hasattr(v.rbr_dense, "weight_rbr_origin"):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, "weight_rbr_avg_conv"): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, "weight_rbr_pfir_conv"):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_idconv1"): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_conv2"):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_dw"):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_pw"):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, "vector"):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    logger.info("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    else:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp["lrf"]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None
    
    # Resume
    start_epoch       = 0
    best_fitness      = 0.0
    best_fitness_p    = 0.0
    best_fitness_r    = 0.0
    best_fitness_f1   = 0.0
    best_fitness_ap50 = 0.0
    best_fitness_ap   = 0.0
    
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness      = ckpt.get("best_fitness",      0.0)
            best_fitness_p    = ckpt.get("best_fitness_p",    0.0)
            best_fitness_r    = ckpt.get("best_fitness_r",    0.0)
            best_fitness_f1   = ckpt.get("best_fitness_f1",   0.0)
            best_fitness_ap50 = ckpt.get("best_fitness_ap50", 0.0)
            best_fitness_ap   = ckpt.get("best_fitness_ap",   0.0)

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]

        # Results
        if ckpt.get("training_results") is not None:
            results_file.write_text(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert start_epoch > 0, "%s training to %g epochs is finished, nothing to resume." % (weights, epochs)
        if epochs < start_epoch:
            logger.info("%s has been trained for %g epochs. Fine-tuning for %g additional epochs." % (weights, ckpt["epoch"], epochs))
            epochs += ckpt["epoch"]  # finetune additional epochs
        
        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.imgsz]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # Trainloader
    dataloader, dataset = create_dataloader(
        path          = train_path,
        imgsz         = imgsz,
        batch_size    = batch_size,
        stride        = gs,
        opt           = opt,
        hyp           = hyp,
        augment       = True,
        cache         = opt.cache_images,
        rect          = opt.rect,
        rank          = rank,
        world_size    = opt.world_size,
        workers       = opt.workers,
        image_weights = opt.image_weights,
        quad          = opt.quad,
        prefix        = colorstr("train: "),
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb  = len(dataloader)  # number of batches
    assert mlc < nc, "Annotation class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(
            path       = test_path,
            imgsz      = imgsz_test,
            batch_size = batch_size * 2,
            stride     = gs,
            opt        = opt,
            hyp        = hyp,
            cache      = opt.cache_images and not opt.notest,
            rect       = True,
            rank       = -1,
            world_size = opt.world_size,
            workers    = opt.workers,
            pad        = 0.5,
            prefix     = colorstr("val: "),
        )[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c      = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.0  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                # plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram("classes", c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(
            model, 
            device_ids    = [opt.local_rank],
            output_device = opt.local_rank,
            # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
            # find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules(),),
            find_unused_parameters=True,
        )

    # Model parameters
    hyp["box"] *= 3.0 / nl                       # scale to layers
    hyp["cls"] *= nc / 80. * 3.0 / nl            # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3.0 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc    = nc   # attach number of classes to model
    model.hyp   = hyp  # attach hyperparameters to model
    model.gr    = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0      = time.time()
    nw      = max(round(hyp["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps    = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler  = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss     = ComputeLoss(model)  # init loss class
    logger.info(
        f"Image sizes {imgsz} train, {imgsz_test} test\n"
        f"Using {dataloader.num_workers} dataloader workers\n"
        f"Logging results to {save_dir}\n"
        f"Starting training for {epochs} epochs..."
    )
    
    torch.save(model, wdir / "init.pt")
    
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "labels", "img_size"))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni   = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns   = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem   = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s     = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f"train_batch{ni}.jpg"  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in save_dir.glob("train*.jpg") if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=["yaml", "nc", "hyp", "gr", "names", "stride", "class_weights"])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(
                    opt          = opt,
                    data         = data_dict,
                    batch_size   = batch_size * 2,
                    imgsz        = imgsz_test,
                    conf_thres   = opt.conf,
                    iou_thres    = opt.iou,
                    max_det      = opt.max_det,
                    model        = ema.ema,
                    single_cls   = opt.single_cls,
                    dataloader   = testloader,
                    save_dir     = save_dir,
                    verbose      = nc < 50 and final_epoch,
                    plots        = plots and final_epoch,
                    wandb_logger = wandb_logger,
                    compute_loss = compute_loss,
                    is_coco      = is_coco,
                    v5_metric    = opt.v5_metric,
                )
        
            # Update best mAP
            fi      = fitness(np.array(results).reshape(1, -1))         # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_p    = fitness_p(np.array(results).reshape(1, -1))       # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_r    = fitness_r(np.array(results).reshape(1, -1))       # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f1 = fitness_f1(np.array(results).reshape(1, -1))    # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            else:
                fi_f1 = 0.0
            fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))    # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap   = fitness_ap(np.array(results).reshape(1, -1))      # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            
            if fi > best_fitness:
                best_fitness      = fi
            if fi_p > best_fitness_p:
                best_fitness_p    = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r    = fi_r
            if fi_f1 > best_fitness_f1:
                best_fitness_f1   = fi_f1
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap   = fi_ap
            wandb_logger.end_epoch(best_result=best_fitness == fi)
            
            # Write
            with open(results_file, "a") as f:
                f.write(s + "%10.4g" * 7 % results + "\n")  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system("gsutil cp %s gs://%s/results/results%s.txt" % (results_file, opt.bucket, opt.name))
            
            # Log
            tags = [
                "train/box_loss",
                "train/obj_loss",
                "train/cls_loss",  # train loss
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/f1(B)",
                "metrics/map@0.5(B)",
                "metrics/map@0.5-0.95(B)",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",  # val loss
                "x/lr0",
                "x/lr1",
                "x/lr2"  # params
            ]
            results = list(results)
            results.insert(2, fi_f1)
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B
            
            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {
                    "epoch"            : epoch,
                    "best_fitness"     : best_fitness,
                    "best_fitness_p"   : best_fitness_p,
                    "best_fitness_r"   : best_fitness_r,
                    "best_fitness_f1"  : best_fitness_f1,
                    "best_fitness_ap50": best_fitness_ap50,
                    "best_fitness_ap"  : best_fitness_ap,
                    "training_results" : results_file.read_text(),
                    "model"            : deepcopy(model.module if is_parallel(model) else model).half(),
                    "ema"              : deepcopy(ema.ema).half(),
                    "updates"          : ema.updates,
                    "optimizer"        : optimizer.state_dict(),
                    "wandb_id"         : wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
                if best_fitness_p == fi_p:
                    torch.save(ckpt, best_p)
                if best_fitness_r == fi_r:
                    torch.save(ckpt, best_r)
                if best_fitness_f1 == fi_f1:
                    torch.save(ckpt, best_f1)
                if best_fitness_ap50 == fi_ap50:
                    torch.save(ckpt, best_ap50)
                if best_fitness_ap == fi_ap:
                    torch.save(ckpt, best_ap)
                # if epoch == 0:
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                # elif ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                # elif epoch >= (epochs-5):
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ["results.png", "confusion_matrix.png", *[f"{x}_curve.png" for x in ("F1", "PR", "P", "R")]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files if (save_dir / f).exists()]})
        # Test best.pt
        logger.info("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith("coco.yaml") and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else last:  # speed, mAP tests
                results, _, _ = test.test(
                    opt        = opt,
                    data       = opt.data,
                    batch_size = batch_size * 2,
                    imgsz      = imgsz_test,
                    conf_thres = opt.conf,
                    iou_thres  = opt.iou,
                    max_det    = opt.max_det,
                    model      = attempt_load(m, device).half(),
                    single_cls = opt.single_cls,
                    dataloader = testloader,
                    save_dir   = save_dir,
                    save_json  = True,
                    plots      = False,
                    is_coco    = is_coco,
                    v5_metric  = opt.v5_metric,
                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best, best_p, best_r, best_f1, best_ap50, best_ap:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f"gsutil cp {final} gs://{opt.bucket}/weights")  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(
                str(final), 
                type    = "model",
                name    = "run_" + wandb_logger.wandb_run.id + "_model",
                aliases = ["last", "best", "stripped"]
            )
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results

# endregion


# region Main

def main() -> str:
    # Parse arguments
    args        = mon.parse_train_args(model_root=_current_dir)
    model       = mon.Path(args.model)
    model       = model if model.exists() else _current_dir / "config" / "training" / model.name
    model       = str(model.config_file())
    data_       = mon.Path(args.data)
    data_       = data_ if data_.exists() else _current_dir / "data" / data_.name
    data_       = str(data_.config_file())
    hyp         = mon.Path(args.hyp)
    hyp         = hyp if hyp.exists() else _current_dir / "data" / hyp.name
    hyp         = hyp.yaml_file()
    args.model  = model
    args.source = args.data
    args.data   = data_
    args.hyp    = str(hyp)
    
    # Set DDP variables
    args.world_size  = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else  1
    args.global_rank = int(os.environ["RANK"])       if "RANK"       in os.environ else -1
    
    if not args.exist_ok:
        mon.delete_dir(paths=mon.Path(args.save_dir))
    
    set_logging(args.global_rank)
    
    # Resume
    wandb_run = check_wandb_resume(args)
    if args.resume and not wandb_run:  # resume an interrupted run
        ckpt    = args.resume if isinstance(args.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        apriori = args.global_rank, args.local_rank
        with open(mon.Path(ckpt).parent.parent / "args.yaml") as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        args.model, args.weights, args.resume, args.batch_size, args.global_rank, args.local_rank = "", ckpt, True, args.total_batch_size, *apriori
        logger.info("Resuming training from %s" % ckpt)
    else:
        # args.hyp = args.hyp or ("hyp.finetune.yaml" if args.weights else "hyp.scratch.yaml")
        args.data     = check_file(args.data)   # check files
        args.model    = check_file(args.model)  # check files
        args.hyp      = check_file(args.hyp)    # check files
        assert len(args.model) or len(args.weights), "either --cfg or --weights must be specified"
        args.imgsz    = mon.to_list(args.imgsz)
        args.imgsz.extend([args.imgsz[-1]] * (2 - len(args.imgsz)))  # extend to 2 sizes (train, test)
        args.name     = "evolve" if args.evolve else args.name
        args.save_dir = increment_path(mon.Path(args.save_dir), exist_ok=args.exist_ok | args.evolve)  # increment run
    
    # DDP mode
    args.total_batch_size = args.batch_size
    device = select_device(args.device, batch_size=args.batch_size)
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")  # distributed backend
        assert args.batch_size % args.world_size == 0, "--batch-size must be multiple of CUDA device count"
        args.batch_size = args.total_batch_size // args.world_size

    # Hyperparameters
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not args.evolve:
        tb_writer = None  # init loggers
        if args.global_rank in [-1, 0]:
            prefix = colorstr("tensorboard: ")
            logger.info(f"{prefix}Start with 'tensorboard --logdir {args.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(args.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0"            : (1, 1e-5, 1e-1),   # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf"            : (1, 0.01, 1.0),    # final OneCycleLR learning rate (lr0 * lrf)
            "momentum"       : (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay"   : (1, 0.0, 0.001),   # optimizer weight decay
            "warmup_epochs"  : (1, 0.0, 5.0),     # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),    # warmup initial momentum
            "warmup_bias_lr" : (1, 0.0, 0.2),     # warmup initial bias lr
            "box"            : (1, 0.02, 0.2),    # box loss gain
            "cls"            : (1, 0.2, 4.0),     # cls loss gain
            "cls_pw"         : (1, 0.5, 2.0),     # cls BCELoss positive_weight
            "obj"            : (1, 0.2, 4.0),     # obj loss gain (scale with pixels)
            "obj_pw"         : (1, 0.5, 2.0),     # obj BCELoss positive_weight
            "iou_t"          : (0, 0.1, 0.7),     # IoU training threshold
            "anchor_t"       : (1, 2.0, 8.0),     # anchor-multiple threshold
            "anchors"        : (2, 2.0, 10.0),    # anchors per output grid (0 to ignore)
            "fl_gamma"       : (0, 0.0, 2.0),     # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h"          : (1, 0.0, 0.1),     # image HSV-Hue augmentation (fraction)
            "hsv_s"          : (1, 0.0, 0.9),     # image HSV-Saturation augmentation (fraction)
            "hsv_v"          : (1, 0.0, 0.9),     # image HSV-Value augmentation (fraction)
            "degrees"        : (1, 0.0, 45.0),    # image rotation (+/- deg)
            "translate"      : (1, 0.0, 0.9),     # image translation (+/- fraction)
            "scale"          : (1, 0.0, 0.9),     # image scale (+/- gain)
            "shear"          : (1, 0.0, 10.0),    # image shear (+/- deg)
            "perspective"    : (0, 0.0, 0.001),   # image perspective (+/- fraction), range 0-0.001
            "flipud"         : (1, 0.0, 1.0),     # image flip up-down (probability)
            "fliplr"         : (0, 0.0, 1.0),     # image flip left-right (probability)
            "mosaic"         : (1, 0.0, 1.0),     # image mixup (probability)
            "mixup"          : (1, 0.0, 1.0),     # image mixup (probability)
            "copy_paste"     : (1, 0.0, 1.0),     # segment copy-paste (probability)
            "paste_in"       : (1, 0.0, 1.0),     # segment copy-paste (probability)
        }    
        
        with open(args.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)   # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
                
        assert args.local_rank == -1, "DDP mode not implemented for --evolve"
        args.notest, args.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = mon.Path(args.save_dir) / "hyp_evolved.yaml"  # save best result here
        if args.bucket:
            os.system("gsutil cp gs://%s/evolve.txt ." % args.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if mon.Path("evolve.txt").exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x      = np.loadtxt("evolve.txt", ndmin=2)
                n      = min(5, len(x))  # number of previous results to consider
                x      = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w      = fitness(x) - fitness(x).min()   # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]    # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()     # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr   = np.random
                npr.seed(int(time.time()))
                g     = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng    = len(meta)
                v     = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):   # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)   # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, args.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}"
        )
        
        return str(args.save_dir)
        

if __name__ == "__main__":
    main()
    
# endregion
