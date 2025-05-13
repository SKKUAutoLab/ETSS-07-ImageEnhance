#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predicts model output on a given dataset."""

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"] or "auto"
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
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red] {data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    args["modelmodule"] |= {
        "fullname" : fullname,
        "root"     : save_dir,
        "weights"  : weights,
        # "optimizer": None,  # Skip initialization for efficiency
        # "loss"     : None,  # Skip initialization for efficiency
        # "metrics"  : None,  # Skip initialization for efficiency
        "debug"    : save_debug,
        "verbose"  : verbose,
    }
    model: mon.Model = mon.MODELS.build(config=args["modelmodule"])
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark and hasattr(model, "compute_efficiency_score"):
        flops, params = model.compute_efficiency_score(image_size=imgsz)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
        
    # Predicting
    run_time = []
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            
            # Infer
            outputs = model.infer(datapoint=datapoint, image_size=imgsz, resize=resize)
            time    = outputs.pop("time", None)
            if time:
                run_time.append(time)
            
            # Save image
            if save_image:
                _, output   = outputs.popitem()
                output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                output_path = output_dir / "predict" / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                mon.write_image(output_path, output)
            
            # Save Debug
            if save_debug:
                debug_dir = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                for k, v in outputs.items():
                    if mon.is_image(v):
                        path = debug_dir / "debug" / f"{image_path.stem}_{k}{mon.SAVE_IMAGE_EXT}"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        mon.write_image(path, v)
    
    # Finish
    avg_time = float(sum(run_time) / len(run_time)) if run_time else 0
    mon.console.log(f"Average time: {avg_time}")
    return str(save_dir)


# ----- Main -----
def main():
    args = mon.parse_predict_args()
    predict(args)


if __name__ == "__main__":
    main()
