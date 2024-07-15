#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

import mon
from utils.general import (
    strip_optimizer,
)

logger        = logging.getLogger(__name__)
console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Post-processing

def main():
    for f in [
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best_ap.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best_ap50.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best_f1.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best_p.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best_r.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/best.pt",
        # "/home/longpham/10-workspace/11-code/mon/project/aic24_fisheye8k/run/train/yolov7_e6e_aic24_fisheye8k_1536/weights/last.pt",
    ]:
        f = mon.Path(f)
        if f.exists():
            strip_optimizer(f)  # strip optimizers

# endregion


# region Main

if __name__ == "__main__":
    main()
    
# endregion
