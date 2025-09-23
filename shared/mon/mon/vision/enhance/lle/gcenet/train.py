#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCE-Net model training pipeline for low-light image enhancement."""

import box
import torch

import mon
import gcenet

mon.dev()

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


def train(args: dict | box.Box) -> str:
     # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")

    # Model
    args.network |= {
        "name"   : args.model,
        "scale"  : 1,
        "weights": pretrained,
    }
    model = mon.MODELS.build(**args.network)
    if pretrained and pretrained.is_weights_file(exist=True):
        model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = mon.nn.Adam(model.parameters(), **args.optimizer)
    
    # Loss
    L_tv    = gcenet.L_tv().to(device)
    L_spa   = gcenet.L_spa().to(device)
    L_col   = gcenet.L_col().to(device)
    L_exp   = gcenet.L_exp(16, args.loss.L_exp_mean).to(device)
    L_tv_w  = args.loss.L_tv_w
    L_spa_w = args.loss.L_spa_w
    L_col_w = args.loss.L_col_w
    L_exp_w = args.loss.L_exp_w
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Train
    best_loss      = 9999
    best_psnr      = 0
    grad_clip_norm = args["trainer"]["grad_clip_norm"]
    with mon.create_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            loss_li  = []
            val_psnr = []
            model.train()
            for j, datapoint in enumerate(train_dataloader):
                image    = datapoint["image"]
                image    = image.to(device)
                depth    = datapoint.get("depth", None)
                depth    = depth.to(device) if depth is not None else None
                outputs  = model(image, depth)
                r        = outputs[0]
                z        = outputs[1]
                enhanced = outputs[-1]
                
                l_tv  = L_tv_w  * L_tv(r)
                l_spa = L_spa_w * torch.mean(L_spa(enhanced, image))
                l_col = L_col_w * torch.mean(L_col(enhanced))
                l_exp = L_exp_w * torch.mean(L_exp(enhanced))
                loss  = l_tv + l_spa + l_col + l_exp
                
                optimizer.zero_grad()
                loss.backward()
                mon.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                loss_li.append(loss.item())
            mean_loss = sum(loss_li) / len(loss_li)
            
            # Validation
            model.eval()
            for j, datapoint in enumerate(val_dataloader):
                with torch.no_grad():
                    image    = datapoint["image"]
                    image    = image.to(device)
                    depth    = datapoint.get("depth", None)
                    depth    = depth.to(device) if depth is not None else None
                    ref      = datapoint["ref"]
                    ref      = ref.to(device)
                    outputs  = model(image, depth)
                    enhanced = outputs[-1]
                    mse      = ((enhanced - ref) ** 2).mean((2, 3))
                    psnr     = (1 / mse).log10().mean() * 10
                val_psnr.append(psnr.item())
            mean_psnr = sum(val_psnr) / len(val_psnr)
            
            # Log
            if args.verbose:
                mon.log(f"Epoch: {(i + 1):03} | Train Loss: {mean_loss:08.6f} | Val PSNR: {mean_psnr:08.6f}")
                
            # Save
            torch.save(model.state_dict(), args.save_dir / "last.pt")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), args.save_dir / "best_loss.pt")
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save(model.state_dict(), args.save_dir / "best_psnr.pt")
            
    # Save last model
    if args.model == "gcenet_mobileone":
        reparam_model = gcenet.reparameterize_model(model)
        torch.save(reparam_model.state_dict(), args.save_dir / "last.pt")
    

# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
