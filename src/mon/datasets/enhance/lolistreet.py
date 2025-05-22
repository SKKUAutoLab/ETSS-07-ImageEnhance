#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LoLI-Street datasets."""

__all__ = [
    "LoLIStreet",
    "LoLIStreetDataModule",
    "LoLIStreetTest",
    "LoLIStreetVal",
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
@DATASETS.register(name="lolistreet")
class LoLIStreet(VisionDataset):
    """Loads LoLI-Street dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    classlabels         = ClassLabels([
        {"id": 0 , "name": "person"        , "supercategory": "person",     "color": [ 81, 120, 228]},
        {"id": 1 , "name": "bicycle"       , "supercategory": "vehicle",    "color": [138, 183,  33]},
        {"id": 2 , "name": "car"           , "supercategory": "vehicle",    "color": [ 49,   3, 150]},
        {"id": 3 , "name": "motorcycle"    , "supercategory": "vehicle",    "color": [122,  35,   2]},
        {"id": 4 , "name": "airplane"      , "supercategory": "vehicle",    "color": [165, 168, 193]},
        {"id": 5 , "name": "bus"           , "supercategory": "vehicle",    "color": [140,  24, 143]},
        {"id": 6 , "name": "train"         , "supercategory": "vehicle",    "color": [179, 165, 212]},
        {"id": 7 , "name": "truck"         , "supercategory": "vehicle",    "color": [ 72, 153, 152]},
        {"id": 8 , "name": "boat"          , "supercategory": "vehicle",    "color": [ 19,  64,  83]},
        {"id": 9 , "name": "traffic light" , "supercategory": "outdoor",    "color": [122,  40,  57]},
        {"id": 10, "name": "fire hydrant"  , "supercategory": "outdoor",    "color": [219,  42, 205]},
        {"id": 11, "name": "stop sign"     , "supercategory": "outdoor",    "color": [ 15,  90, 125]},
        {"id": 12, "name": "parking meter" , "supercategory": "outdoor",    "color": [187,  80,  10]},
        {"id": 13, "name": "bench"         , "supercategory": "outdoor",    "color": [ 76, 226, 142]},
        {"id": 14, "name": "bird"          , "supercategory": "animal",     "color": [ 24,  56,  34]},
        {"id": 15, "name": "cat"           , "supercategory": "animal",     "color": [ 41, 174, 251]},
        {"id": 16, "name": "dog"           , "supercategory": "animal",     "color": [ 21,   8, 251]},
        {"id": 17, "name": "horse"         , "supercategory": "animal",     "color": [106, 128, 177]},
        {"id": 18, "name": "sheep"         , "supercategory": "animal",     "color": [147,  90, 131]},
        {"id": 19, "name": "cow"           , "supercategory": "animal",     "color": [ 65, 159, 189]},
        {"id": 20, "name": "elephant"      , "supercategory": "animal",     "color": [129,  70,  30]},
        {"id": 21, "name": "bear"          , "supercategory": "animal",     "color": [ 38, 181,  29]},
        {"id": 22, "name": "zebra"         , "supercategory": "animal",     "color": [189, 238, 167]},
        {"id": 23, "name": "giraffe"       , "supercategory": "animal",     "color": [173, 154, 136]},
        {"id": 24, "name": "backpack"      , "supercategory": "accessory",  "color": [205, 104,  95]},
        {"id": 25, "name": "umbrella"      , "supercategory": "accessory",  "color": [163,  13, 178]},
        {"id": 26, "name": "handbag"       , "supercategory": "accessory",  "color": [156,  84, 167]},
        {"id": 27, "name": "tie"           , "supercategory": "accessory",  "color": [ 10, 146, 166]},
        {"id": 28, "name": "suitcase"      , "supercategory": "accessory",  "color": [176, 137,  78]},
        {"id": 29, "name": "frisbee"       , "supercategory": "sports",     "color": [190, 118,  41]},
        {"id": 30, "name": "skis"          , "supercategory": "sports",     "color": [159, 178,  24]},
        {"id": 31, "name": "snowboard"     , "supercategory": "sports",     "color": [107,  85, 171]},
        {"id": 32, "name": "sports ball"   , "supercategory": "sports",     "color": [186, 223, 221]},
        {"id": 33, "name": "kite"          , "supercategory": "sports",     "color": [142, 218,  56]},
        {"id": 34, "name": "baseball bat"  , "supercategory": "sports",     "color": [ 82, 128, 254]},
        {"id": 35, "name": "baseball glove", "supercategory": "sports",     "color": [ 64, 200, 173]},
        {"id": 36, "name": "skateboard"    , "supercategory": "sports",     "color": [112,  66,  51]},
        {"id": 37, "name": "surfboard"     , "supercategory": "sports",     "color": [ 47, 131, 231]},
        {"id": 38, "name": "tennis racket" , "supercategory": "sports",     "color": [ 37,  70, 244]},
        {"id": 39, "name": "bottle"        , "supercategory": "kitchen",    "color": [139, 160,   1]},
        {"id": 40, "name": "wine glass"    , "supercategory": "kitchen",    "color": [103,  32,  74]},
        {"id": 41, "name": "cup"           , "supercategory": "kitchen",    "color": [ 28,  47,  55]},
        {"id": 42, "name": "fork"          , "supercategory": "kitchen",    "color": [219,  18, 203]},
        {"id": 43, "name": "knife"         , "supercategory": "kitchen",    "color": [ 41, 125, 194]},
        {"id": 44, "name": "spoon"         , "supercategory": "kitchen",    "color": [ 76, 180, 131]},
        {"id": 45, "name": "bowl"          , "supercategory": "kitchen",    "color": [143,   4, 187]},
        {"id": 46, "name": "banana"        , "supercategory": "food",       "color": [232, 188,  11]},
        {"id": 47, "name": "apple"         , "supercategory": "food",       "color": [119, 177,  17]},
        {"id": 48, "name": "sandwich"      , "supercategory": "food",       "color": [ 55, 214, 248]},
        {"id": 49, "name": "orange"        , "supercategory": "food",       "color": [100, 254,  62]},
        {"id": 50, "name": "broccoli"      , "supercategory": "food",       "color": [ 15,  12,  37]},
        {"id": 51, "name": "carrot"        , "supercategory": "food",       "color": [105,  24,  82]},
        {"id": 52, "name": "hot dog"       , "supercategory": "food",       "color": [192, 102, 113]},
        {"id": 53, "name": "pizza"         , "supercategory": "food",       "color": [242,  21, 163]},
        {"id": 54, "name": "donut"         , "supercategory": "food",       "color": [ 13,  42, 240]},
        {"id": 55, "name": "cake"          , "supercategory": "food",       "color": [ 83, 228, 215]},
        {"id": 56, "name": "chair"         , "supercategory": "furniture",  "color": [ 94, 173,  36]},
        {"id": 57, "name": "couch"         , "supercategory": "furniture",  "color": [ 63,  48,  10]},
        {"id": 58, "name": "potted plant"  , "supercategory": "furniture",  "color": [199,  53,   7]},
        {"id": 59, "name": "bed"           , "supercategory": "furniture",  "color": [174,  28, 109]},
        {"id": 60, "name": "dining table"  , "supercategory": "furniture",  "color": [216, 147, 179]},
        {"id": 61, "name": "toilet"        , "supercategory": "furniture",  "color": [ 36, 181, 193]},
        {"id": 62, "name": "tv"            , "supercategory": "electronic", "color": [ 54,  95, 132]},
        {"id": 63, "name": "laptop"        , "supercategory": "electronic", "color": [142,  43,  85]},
        {"id": 64, "name": "mouse"         , "supercategory": "electronic", "color": [150, 175,  16]},
        {"id": 65, "name": "remote"        , "supercategory": "electronic", "color": [125, 179, 231]},
        {"id": 66, "name": "keyboard"      , "supercategory": "electronic", "color": [249,  95, 141]},
        {"id": 67, "name": "cell phone"    , "supercategory": "electronic", "color": [105,  24, 191]},
        {"id": 68, "name": "microwave"     , "supercategory": "appliance",  "color": [135,  51,  82]},
        {"id": 69, "name": "oven"          , "supercategory": "appliance",  "color": [ 69,  21,  20]},
        {"id": 70, "name": "toaster"       , "supercategory": "appliance",  "color": [ 67,  30, 125]},
        {"id": 71, "name": "sink"          , "supercategory": "appliance",  "color": [135, 205,  67]},
        {"id": 72, "name": "refrigerator"  , "supercategory": "appliance",  "color": [ 35, 219,  70]},
        {"id": 73, "name": "book"          , "supercategory": "indoor",     "color": [ 80, 203,  31]},
        {"id": 74, "name": "clock"         , "supercategory": "indoor",     "color": [ 26,  26, 253]},
        {"id": 75, "name": "vase"          , "supercategory": "indoor",     "color": [134, 219,  70]},
        {"id": 76, "name": "scissors"      , "supercategory": "indoor",     "color": [  0, 132, 236]},
        {"id": 77, "name": "teddy bear"    , "supercategory": "indoor",     "color": [134,  81,   4]},
        {"id": 78, "name": "hair drier"    , "supercategory": "indoor",     "color": [123,  68, 172]},
        {"id": 79, "name": "toothbrush"    , "supercategory": "indoor",     "color": [ 58, 228, 226]},
    ])

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "lolistreet" if root.name != "lolistreet" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        # if self.split == Split.TEST:
        #     patterns = [self.root / "val" / "image"]
        # else:
        #     patterns = [self.root / self.split_str / "image"]
        patterns = [self.root / self.split_str / "image"]
        
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="lolistreetval")
class LoLIStreetVal(LoLIStreet):
    """Loads LoLI-Street-Val dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "val" / "image"]
        
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

@DATASETS.register(name="lolistreettest")
class LoLIStreetTest(LoLIStreet):
    """Loads LoLI-Street-Test dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "test" / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

# ----- DataModule -----
@DATAMODULES.register(name="lolistreet")
class LoLIStreetDataModule(core.DataModule):
    """Configures LoLI-Street datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = LoLIStreet(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LoLIStreet(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LoLIStreet(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="lolistreetval")
class LoLIStreetValDataModule(core.DataModule):
    """Configures LoLI-Street-Val datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = LoLIStreetVal(split=Split.VAL, **self.dataset_kwargs)
            self.val   = LoLIStreetVal(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LoLIStreetVal(split=Split.VAL, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="lolistreettest")
class LoLIStreetTestDataModule(core.DataModule):
    """Configures LoLI-Street-Test datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = LoLIStreetTest(split=Split.TEST, **self.dataset_kwargs)
            self.val   = LoLIStreetTest(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LoLIStreetTest(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
