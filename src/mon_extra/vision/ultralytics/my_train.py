#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the training scripts for YOLOv8."""

from __future__ import annotations

import mon
import ultralytics.utils
from ultralytics import YOLO

console = mon.console
current_file = mon.Path(__file__).absolute()
_current_dir = current_file.parents[0]

ultralytics.utils.DATASETS_DIR = mon.DATA_DIR


# region Train
def train(args: dict):
    model = YOLO(args["model"])
    _ = model.train(**args)

# endregion


# region Main

def main() -> str:
    # Parse arguments
    args         = mon.parse_train_args(model_root=_current_dir)
    # model        = mon.Path(args.model)
    # model        = model if model.exists() else _current_dir / "config" / model.name
    # model        = str(model.config_file())
    data_        = mon.Path(args.data)
    data_        = data_ if data_.exists() else _current_dir / "data" / data_.name
    data_        = str(data_.config_file())
    hyp          = mon.Path(args.hyp)
    hyp          = hyp if hyp.exists() else _current_dir / "data" / hyp.name
    hyp          = hyp.yaml_file()
    weights      = mon.to_list(args.weights)
    weights      = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    args.mode    = "train"
    args.model   = weights
    args.source  = args.data
    args.data    = data_
    args.hyp     = str(hyp)
    args.project = str(args.save_dir.parent)
    args.name    = str(args.save_dir.name)
    
    train(args=args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
