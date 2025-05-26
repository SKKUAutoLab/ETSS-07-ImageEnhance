#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements COCO 2017 datasets for detection and segmentation."""

__all__ = [
    "COCO80",
    "COCO80DataModule",
]

from typing import Literal

from mon import core, vision
from mon.constants import DATAMODULES, DATASETS, Split, Task

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
InfraredAnnotation             = vision.InfraredAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="coco80")
class COCO80(VisionDataset):
    """COCO-80-classes dataset."""
    
    tasks : list[Task]  = [Task.DETECT]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        # "bbox" : BBoxesAnnotation,
    })
    has_test_annotations: bool = False
    classlabels         : ClassLabels = core.ClassLabels([
        {"name": "background"    , "id":  0, "supercategory": "background", "color": [  0,   0,   0]},
        {"name": "person"        , "id":  1, "supercategory": "person"    , "color": [ 81, 120, 228]},
        {"name": "bicycle"       , "id":  2, "supercategory": "vehicle"   , "color": [138, 183,  33]},
        {"name": "car"           , "id":  3, "supercategory": "vehicle"   , "color": [ 49,   3, 150]},
        {"name": "motorcycle"    , "id":  4, "supercategory": "vehicle"   , "color": [122,  35,   2]},
        {"name": "airplane"      , "id":  5, "supercategory": "vehicle"   , "color": [165, 168, 193]},
        {"name": "bus"           , "id":  6, "supercategory": "vehicle"   , "color": [140,  24, 143]},
        {"name": "train"         , "id":  7, "supercategory": "vehicle"   , "color": [179, 165, 212]},
        {"name": "truck"         , "id":  8, "supercategory": "vehicle"   , "color": [ 72, 153, 152]},
        {"name": "boat"          , "id":  9, "supercategory": "vehicle"   , "color": [ 19,  64,  83]},
        {"name": "traffic light" , "id": 10, "supercategory": "outdoor"   , "color": [122,  40,  57]},
        {"name": "fire hydrant"  , "id": 11, "supercategory": "outdoor"   , "color": [219,  42, 205]},
        {"name": "stop sign"     , "id": 12, "supercategory": "outdoor"   , "color": [ 15,  90, 125]},
        {"name": "parking meter" , "id": 13, "supercategory": "outdoor"   , "color": [187,  80,  10]},
        {"name": "bench"         , "id": 14, "supercategory": "outdoor"   , "color": [ 76, 226, 142]},
        {"name": "bird"          , "id": 15, "supercategory": "animal"    , "color": [ 24,  56,  34]},
        {"name": "cat"           , "id": 16, "supercategory": "animal"    , "color": [ 41, 174, 251]},
        {"name": "dog"           , "id": 17, "supercategory": "animal"    , "color": [ 21,   8, 251]},
        {"name": "horse"         , "id": 18, "supercategory": "animal"    , "color": [106, 128, 177]},
        {"name": "sheep"         , "id": 19, "supercategory": "animal"    , "color": [147,  90, 131]},
        {"name": "cow"           , "id": 20, "supercategory": "animal"    , "color": [ 65, 159, 189]},
        {"name": "elephant"      , "id": 21, "supercategory": "animal"    , "color": [129,  70,  30]},
        {"name": "bear"          , "id": 22, "supercategory": "animal"    , "color": [ 38, 181,  29]},
        {"name": "zebra"         , "id": 23, "supercategory": "animal"    , "color": [189, 238, 167]},
        {"name": "giraffe"       , "id": 24, "supercategory": "animal"    , "color": [173, 154, 136]},
        {"name": "backpack"      , "id": 25, "supercategory": "accessory" , "color": [205, 104,  95]},
        {"name": "umbrella"      , "id": 26, "supercategory": "accessory" , "color": [163,  13, 178]},
        {"name": "handbag"       , "id": 27, "supercategory": "accessory" , "color": [156,  84, 167]},
        {"name": "tie"           , "id": 28, "supercategory": "accessory" , "color": [ 10, 146, 166]},
        {"name": "suitcase"      , "id": 29, "supercategory": "accessory" , "color": [176, 137,  78]},
        {"name": "frisbee"       , "id": 30, "supercategory": "sports"    , "color": [190, 118,  41]},
        {"name": "skis"          , "id": 31, "supercategory": "sports"    , "color": [159, 178,  24]},
        {"name": "snowboard"     , "id": 32, "supercategory": "sports"    , "color": [107,  85, 171]},
        {"name": "sports ball"   , "id": 33, "supercategory": "sports"    , "color": [186, 223, 221]},
        {"name": "kite"          , "id": 34, "supercategory": "sports"    , "color": [142, 218,  56]},
        {"name": "baseball bat"  , "id": 35, "supercategory": "sports"    , "color": [ 82, 128, 254]},
        {"name": "baseball glove", "id": 36, "supercategory": "sports"    , "color": [ 64, 200, 173]},
        {"name": "skateboard"    , "id": 37, "supercategory": "sports"    , "color": [112,  66,  51]},
        {"name": "surfboard"     , "id": 38, "supercategory": "sports"    , "color": [ 47, 131, 231]},
        {"name": "tennis racket" , "id": 39, "supercategory": "sports"    , "color": [ 37,  70, 244]},
        {"name": "bottle"        , "id": 40, "supercategory": "kitchen"   , "color": [139, 160,   1]},
        {"name": "wine glass"    , "id": 41, "supercategory": "kitchen"   , "color": [103,  32,  74]},
        {"name": "cup"           , "id": 42, "supercategory": "kitchen"   , "color": [ 28,  47,  55]},
        {"name": "fork"          , "id": 43, "supercategory": "kitchen"   , "color": [219,  18, 203]},
        {"name": "knife"         , "id": 44, "supercategory": "kitchen"   , "color": [ 41, 125, 194]},
        {"name": "spoon"         , "id": 45, "supercategory": "kitchen"   , "color": [ 76, 180, 131]},
        {"name": "bowl"          , "id": 46, "supercategory": "kitchen"   , "color": [143,   4, 187]},
        {"name": "banana"        , "id": 47, "supercategory": "food"      , "color": [232, 188,  11]},
        {"name": "apple"         , "id": 48, "supercategory": "food"      , "color": [119, 177,  17]},
        {"name": "sandwich"      , "id": 49, "supercategory": "food"      , "color": [ 55, 214, 248]},
        {"name": "orange"        , "id": 50, "supercategory": "food"      , "color": [100, 254,  62]},
        {"name": "broccoli"      , "id": 51, "supercategory": "food"      , "color": [ 15,  12,  37]},
        {"name": "carrot"        , "id": 52, "supercategory": "food"      , "color": [105,  24,  82]},
        {"name": "hot dog"       , "id": 53, "supercategory": "food"      , "color": [192, 102, 113]},
        {"name": "pizza"         , "id": 54, "supercategory": "food"      , "color": [242,  21, 163]},
        {"name": "donut"         , "id": 55, "supercategory": "food"      , "color": [ 13,  42, 240]},
        {"name": "cake"          , "id": 56, "supercategory": "food"      , "color": [ 83, 228, 215]},
        {"name": "chair"         , "id": 57, "supercategory": "furniture" , "color": [ 94, 173,  36]},
        {"name": "couch"         , "id": 58, "supercategory": "furniture" , "color": [ 63,  48,  10]},
        {"name": "potted plant"  , "id": 59, "supercategory": "furniture" , "color": [199,  53,   7]},
        {"name": "bed"           , "id": 60, "supercategory": "furniture" , "color": [174,  28, 109]},
        {"name": "dining table"  , "id": 61, "supercategory": "furniture" , "color": [216, 147, 179]},
        {"name": "toilet"        , "id": 62, "supercategory": "furniture" , "color": [ 36, 181, 193]},
        {"name": "tv"            , "id": 63, "supercategory": "electronic", "color": [ 54,  95, 132]},
        {"name": "laptop"        , "id": 64, "supercategory": "electronic", "color": [142,  43,  85]},
        {"name": "mouse"         , "id": 65, "supercategory": "electronic", "color": [150, 175,  16]},
        {"name": "remote"        , "id": 66, "supercategory": "electronic", "color": [125, 179, 231]},
        {"name": "keyboard"      , "id": 67, "supercategory": "electronic", "color": [249,  95, 141]},
        {"name": "cell phone"    , "id": 68, "supercategory": "electronic", "color": [105,  24, 191]},
        {"name": "microwave"     , "id": 69, "supercategory": "appliance" , "color": [135,  51,  82]},
        {"name": "oven"          , "id": 70, "supercategory": "appliance" , "color": [ 69,  21,  20]},
        {"name": "toaster"       , "id": 71, "supercategory": "appliance" , "color": [ 67,  30, 125]},
        {"name": "sink"          , "id": 72, "supercategory": "appliance" , "color": [135, 205,  67]},
        {"name": "refrigerator"  , "id": 73, "supercategory": "appliance" , "color": [ 35, 219,  70]},
        {"name": "book"          , "id": 74, "supercategory": "indoor"    , "color": [ 80, 203,  31]},
        {"name": "clock"         , "id": 75, "supercategory": "indoor"    , "color": [ 26,  26, 253]},
        {"name": "vase"          , "id": 76, "supercategory": "indoor"    , "color": [134, 219,  70]},
        {"name": "scissors"      , "id": 77, "supercategory": "indoor"    , "color": [  0, 132, 236]},
        {"name": "teddy bear"    , "id": 78, "supercategory": "indoor"    , "color": [134,  81,   4]},
        {"name": "hair drier"    , "id": 79, "supercategory": "indoor"    , "color": [123,  68, 172]},
        {"name": "toothbrush"    , "id": 80, "supercategory": "indoor"    , "color": [ 58, 228, 226]},
    ])
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "coco2017" if root.name != "coco2017" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")

        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / self.split_str / "image",]
        
        # Left Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
					sequence    = sorted(list(pattern.rglob("*"))),
					description = f"Listing {self.__class__.__name__} {self.split_str} images"
				):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path))
        
        self.datapoints["image"] = images


# ----- DataModule -----
@DATAMODULES.register(name="coco80")
class COCO80DataModule(core.DataModule):
    
    tasks: list[Task] = [Task.DETECT]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = COCO80(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = COCO80(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = COCO80(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
