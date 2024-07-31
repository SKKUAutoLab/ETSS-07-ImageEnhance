#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import mon
import ultralytics.utils
from ultralytics import YOLO

console = mon.console
current_file = mon.Path(__file__).absolute()
_current_dir = current_file.parents[0]

ultralytics.utils.DATASETS_DIR = mon.DATA_DIR


# region Predict
def predict(args: dict):
    model = YOLO(args["model"])
    _project = args.pop("project")
    _name = args.pop("name")
    project = f"{_project}/{_name}"
    sources = args.pop("source")
    sources = [sources] if not isinstance(sources, list) else sources
    for source in sources:
        path = mon.Path(source)
        name = path.parent.name if path.name == "images" else path.name
        _ = model(source=source, project=f"{project}", name=name, **args)


# endregion


# region Main

def main() -> str:
    # Parse args
    args         = mon.parse_predict_args(model_root=_current_dir)
    args.mode    = "predict"
    args.model   = args.weights
    args.project = str(args.save_dir.parent)
    args.name    = str(args.save_dir.name)
    args.batch   = 1
    args.source  = args.data
    
    predict(args=vars(args))
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
