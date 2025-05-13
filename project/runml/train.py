#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a model on a given dataset."""

import mon

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
    # imgsz        = args["imgsz"]
    # resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]

    # Start
    if mon.is_rank_zero():
        mon.console.rule("[bold red] INITIALIZATION")
        mon.console.log(f"Machine: {hostname}")
    
    # Device
    # device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(root, args["datamodule"].get("root", "")),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(stage="train")
    
    # Model
    args["modelmodule"] |= {
        "fullname": fullname,
        "root"    : save_dir,
        "weights" : weights,
        "debug"   : save_debug,
        "verbose" : verbose,
    }
    model: mon.Model = mon.MODELS.build(config=args["modelmodule"])
    if mon.is_rank_zero():
        mon.print_dict(args, title=fullname)
        mon.console.log("[green]Done")
    
    # Trainer
    if mon.is_rank_zero():
        mon.console.rule("[bold red] SETUP TRAINER")
    
    callbacks = args["trainer"]["callbacks"]
    for i, callback in enumerate(callbacks):
        if callback["name"] == "model_checkpoint":
            callbacks[i] |= {"filename": fullname}
    callbacks = mon.CALLBACKS.build_instances(configs=args["trainer"]["callbacks"])
    ckpt      = mon.get_latest_checkpoint(dirpath=model.ckpt_dir)
    devices   = mon.to_int_list(device) if "auto" not in device else "auto"
    if args["trainer"]["logger"]:
        logger = [mon.TensorBoardLogger(save_dir=save_dir)]
    else:
        logger = False
    
    args["trainer"] |= {
        "callbacks"           : callbacks,
        "devices"             : devices,
        "default_root_dir"    : save_dir,
        "logger"              : logger,
        "max_epochs"          : epochs,
        "max_steps"           : steps,
        "num_sanity_val_steps": 0,
    }
    trainer               = mon.Trainer(**args["trainer"])
    trainer.current_epoch = mon.get_epoch_from_checkpoint(ckpt=ckpt)
    trainer.global_step   = mon.get_global_step_from_checkpoint(ckpt=ckpt)
    if mon.is_rank_zero():
        mon.console.log("[green]Done")
    
    # Training
    if mon.is_rank_zero():
        mon.console.rule("[bold red] TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )
    if mon.is_rank_zero():
        mon.console.log(f"Model: {fullname}")  # Log
        mon.console.log("[green]Done")
    
    # Return
    return str(save_dir)
    

# ----- Main -----
def main():
    args = mon.parse_train_args()
    train(args)


if __name__ == "__main__":
    main()
