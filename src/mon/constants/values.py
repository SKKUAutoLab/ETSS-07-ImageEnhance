#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines scalar constants."""

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "DETECTORS",
    "DISTANCES",
    "EMBEDDERS",
    "EXTRA_DATASETS",
    "EXTRA_MODELS",
    "EXTRA_STR",
    "Enum",
    "LOGGERS",
    "LOSSES",
    "LR_SCHEDULERS",
    "METRICS",
    "MODELS",
    "MON_DIR",
    "MON_EXTRA_DIR",
    "MOTIONS",
    "OBJECTS",
    "OPTIMIZERS",
    "ROOT_DIR",
    "SAVE_CKPT_EXT",
    "SAVE_DEBUG_DIR",
    "SAVE_IMAGE_DIR",
    "SAVE_IMAGE_EXT",
    "SAVE_LABEL_DIR",
    "SAVE_VISUALIZE_DIR",
    "SAVE_WEIGHTS_EXT",
    "SERIALIZERS",
    "STRATEGIES",
    "TRACKERS",
    "TRANSFORMS",
    "ZOO_DIR",
]

from mon.constants.enums import *
from mon.core import factory, pathlib


# ----- Directory -----
current_file  = pathlib.Path(__file__).absolute()
ROOT_DIR      = current_file.parents[3]     # ./mon
DATA_DIR      = ROOT_DIR / "data"           # ./mon/data
SRC_DIR       = ROOT_DIR / "src"            # ./mon/src
MON_DIR       = ROOT_DIR / "src/mon"        # ./mon/src/mon
MON_EXTRA_DIR = ROOT_DIR / "src/mon/extra"  # ./mon/src/mon/extra
ZOO_DIR       = ROOT_DIR / "zoo"            # ./mon/zoo

'''
ZOO_DIR = None
for i, parent in enumerate(current_file.parents):
    if (parent / "zoo").is_dir():
        ZOO_DIR = parent / "zoo"
        break
    if i >= 5:
        break
if ZOO_DIR is None:
    raise Warning(f"Cannot locate the ``zoo`` directory.")

DATA_DIR = os.getenv("DATA_DIR", None)
DATA_DIR = pathlib.Path(DATA_DIR) if DATA_DIR else None
DATA_DIR = DATA_DIR or pathlib.Path("/data")
DATA_DIR = DATA_DIR if DATA_DIR.is_dir() else ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    raise Warning(f"Cannot locate the ``data`` directory.")
'''


# ----- Constants -----
SAVE_DEBUG_DIR     = "debug"
SAVE_IMAGE_DIR     = "pred"
SAVE_LABEL_DIR     = "label"
SAVE_VISUALIZE_DIR = "visualize"
SAVE_CKPT_EXT      = TorchExtension.CKPT.value
SAVE_IMAGE_EXT     = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT   = TorchExtension.PT.value
# List 3rd party modules
EXTRA_STR      = "[extra]"
EXTRA_DATASETS = {}
EXTRA_MODELS   = {  # architecture/model (+ variant)
    # region detect
    "deim"  : {
        "deim_dfine_l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r18vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r34vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r50vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r50vd_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r101vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
    },
    "dfine" : {
        "dfine_l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
    },
    "yolor" : {
        "yolor_d6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_e6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_p6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_w6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
    },
    "yolov7": {
        "yolov7"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_d6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6e": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_w6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7x"   : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
    },
    "yolov8": {
        "yolov8n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
    },
    "yolov9": {
        "gelan_c" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "gelan_e" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_c": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_e": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
    },
    # endregion
    # region enhance/dehaze
    "zid": {
        "zid": {
            "tasks"    : [Task.DEHAZE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "dehaze" / "zid",
        },
    },
    # endregion
    # region enhance/demoire
    "esdnet": {
        "esdnet"  : {
            "tasks"    : [Task.DEMOIRE, Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
        "esdnet_l": {
            "tasks"    : [Task.DEMOIRE, Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
    },
    # endregion
    # region enhance/derain
    "esdnet_snn": {
        "esdnet_snn": {
            "tasks"    : [Task.DERAIN, Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "derain" / "esdnet_snn",
        },
    },
    # endregion
    # region enhance/lle
    "colie"           : {
        "colie": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "colie",
        },
    },
    "dccnet"          : {
        "dccnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "dccnet",
        },
    },
    "enlightengan"    : {
        "enlightengan": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "enlightengan",
        },
    },
    "fourllie"        : {
        "fourllie": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "fourllie",
        },
    },
    "hvi_cidnet"      : {
        "hvi_cidnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "hvi_cidnet",
        },
    },
    "li2025"          : {
        "li2025": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "li2025",
        },
    },
    "lightendiffusion": {
        "lightendiffusion": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "lightendiffusion",
        },
    },
    "lime"            : {
        "lime": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "lime",
        },
    },
    "llflow"          : {
        "llflow": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "llflow",
        },
    },
    "llunet++"        : {
        "llunet++": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "llunet++",
        },
    },
    "nerco"           : {
        "nerco": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "nerco",
        },
    },
    "pairlie"         : {
        "pairlie": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "pairlie",
        },
    },
    "pie"             : {
        "pie": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "pie",
        },
    },
    "psenet"          : {
        "psenet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "psenet",
        },
    },
    "quadprior"       : {
        "quadprior": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "quadprior",
        }
    },
    "retinexformer"   : {
        "retinexformer": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "retinexformer",
        },
    },
    "retinexnet"      : {
        "retinexnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "retinexnet",
        },
    },
    "rsfnet"          : {
        "rsfnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "rsfnet",
        },
    },
    "ruas"            : {
        "ruas": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "ruas",
        },
    },
    "sci"             : {
        "sci": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "sci",
        },
    },
    "sgz"             : {
        "sgz": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "sgz",
        },
    },
    "snr"             : {
        "snr": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "snr",
        },
    },
    "uretinexnet"     : {
        "uretinexnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "uretinexnet",
        },
    },
    "uretinexnet++"   : {
        "uretinexnet++": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "uretinexnet++",
        },
    },
    "utvnet"          : {
        "utvnet": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "utvnet",
        },
    },
    "zerodce"         : {
        "zerodce"  : {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "zerodce",
        },
    },
    "zerodce++"       : {
        "zerodce++": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "zerodce++",
        },
    },
    "zerodidce"       : {
        "zerodidce": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "zerodidce",
        },
    },
    "zeroig"          : {
        "zeroig": {
            "tasks"    : [Task.LLE],
            "mltypes"  : [MLType.ZERO_SHOT, MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "zeroig",
        },
    },
    "zerotig"         : {
        "zerotig": {
            "tasks"    : [Task.LLE, Task.VIDEO],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "lle" / "zerotig",
        },
    },
    # endregion
    # region enhance/multitask
    "airnet"   : {
        "airnet": {
            "tasks"    : [Task.DENOISE, Task.DERAIN, Task.DEHAZE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "airnet",
        },
    },
    "restormer": {
        "restormer": {
            "tasks"    : [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW, Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "restormer",
        },
    },
    # endregion
    # region enhance/retouch
    "neurop": {
        "neurop": {
            "tasks"    : [Task.RETOUCH, Task.LLE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "retouch" / "neurop",
        },
    },
    # endregion
    # region enhance/rr
    "rdnet": {
        "rdnet": {
            "tasks"    : [Task.RR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "rr" / "rdnet",
        },
    },
    # endregion
    # region enhance/sr
    "sronet": {
        "sronet": {
            "tasks"    : [Task.SR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "sr" / "sronet",
        },
    },
    # endregion
    # region segment
    "sam" : {
        "sam_vit_b": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_h": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_l": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
    },
    "sam2": {
        "sam2_hiera_b+": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_l" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_s" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_t" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
    },
    # endregion
    # region types/depth
    "depth_anything_v2": {
        "depth_anything_v2_vitb": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vits": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitl": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitg": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
    },
    "depth_pro"        : {
        "depth_pro": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_pro",
        },
    },
    # endregion
    # region types/thermal
    "srgb_tir": {
        "srgb_tir": {
            "tasks"    : [Task.RGB2TIR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "thermal" / "srgb_tir",
        },
    },
    # endregion
}


# ----- Factory -----
ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
DETECTORS     = factory.Factory(name="Detectors")
DISTANCES     = factory.Factory(name="Distances")
EMBEDDERS     = factory.Factory(name="Embedders")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.ModelFactory(name="Models")
MOTIONS       = factory.Factory(name="Motions")
OBJECTS       = factory.Factory(name="Objects")
OPTIMIZERS    = factory.Factory(name="Optimizers")
SERIALIZERS   = factory.Factory(name="Serializers")
STRATEGIES    = factory.Factory(name="Strategies")
TRACKERS      = factory.Factory(name="Trackers")
TRANSFORMS    = factory.Factory(name="Transforms")
