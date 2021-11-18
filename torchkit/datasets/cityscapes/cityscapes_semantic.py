#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes semantic segmentation dataset and datamodule.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
from torchkit.core.data import ClassLabels
from torchkit.core.data.image_info import ImageInfo
from torchkit.core.data import VisionData
from torchkit.core.dataset import CityscapesLabelHandler
from torchkit.core.dataset import DataModule
from torchkit.core.dataset import SemanticDataset
from torchkit.core.dataset import VisualDataHandler
from torchkit.core.image import is_image_file
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchkit.core.utils import Dim3
from torchkit.core.utils import unique

from torchkit.datasets.builder import DATAMODULES
from torchkit.datasets.builder import DATASETS
from torchkit.utils import datasets_dir
from torchkit.utils import load_config

logger = logging.getLogger()

__all__ = ["CityscapesSemantic", "CityscapesSemanticDataModule"]


# MARK: - Data Config

data = {
    "name": "cityscapes_semantic",
    # The dataset's name.
    "quality": "gtFine",
    # The quality of the semantic segmentation mask to use. One of:
    # [`gtFine`, `gtCoarse`]. Default: `gtFine`.
    "encoding": "id",
    # The format to use when creating the semantic segmentation mask.
    # One of: [`id`, `trainId`, `catId`, `color`]. Default: `id`.
    "extra": False,
    # Should use extra data? Those in the `train_extra` split are only
    # available for `quality=gtCoarse`. Default: `False`.
    "shape": [1024, 2048, 3],
    # The image shape as [H, W, C]. This is compatible with OpenCV format.
    # This is also used to reshape the input data.
    "num_classes": 34,
    # The number of classes in the dataset.
    "batch_size": 4,
    # Number of samples in one forward & backward pass.
    "label_format": "custom",
    # The format to convert images and labels to when `get_item()`. Each
    # labels' format has each own directory: annotations_<format>.
    # Example: `.../train/annotations_yolo/...`
    # Supports:
    # - `custom`: uses our custom annotation format.
    # - `coco`  : uses Coco annotation format.
    # - `yolo`  : uses Yolo annotation format.
    # - `pascal`: uses Pascal annotation format.
    # Default: `yolo`.
    "caching_labels": False,
    # Should overwrite the existing cached labels? Default: `False`.
    "caching_images": False,
    # Cache images into memory for faster training. Default: `False`.
    "write_labels": False,
    # After loading images and labels for the first time, we will convert it to
    # our custom data format and write to files. If `True`, we will overwrite
    # these files. Default: `False`.
    "fast_dev_run": False,
    # Take a small subset of the data for fast debug (i.e, like unit testing).
    # Default: `False`.
    "shuffle": True,
    # Set to `True` to have the data reshuffled at every training epoch.
    # Default: `True`.
    "augment": {
        "hsv_h": 0.015,  # Image HSV-Hue augmentation (fraction).
        "hsv_s": 0.7,  # Image HSV-Saturation augmentation (fraction).
        "hsv_v": 0.4,  # Image HSV-Value augmentation (fraction).
        "rotate": 0.0,  # Image rotation (+/- deg).
        "translate": 0.5,  # Image translation (+/- fraction).
        "scale": 0.5,  # Image scale (+/- gain).
        "shear": 0.0,  # Image shear (+/- deg).
        "perspective": 0.0,  # Image perspective (+/- fraction), range 0-0.001.
        "flip_ud": 0.0,  # Image flip up-down (probability).
        "flip_lr": 0.5,  # Image flip left-right (probability).
        "mixup": 0.0,  # Image mixup (probability).
        "mosaic": True,  # Use mosaic augmentation.
        "rect": False,
        # Train model using rectangular images instead of square ones.
        "stride": 32,
        # When `rect_training=True`, reshape the image shapes as a multiply of
        # stride.
        "pad": 0.0,
        # When `rect_training=True`, pad the empty pixel with given values
    },
}


# MARK: - CityscapesSemantic

@DATASETS.register(name="cityscapes")
@DATASETS.register(name="cityscapes_semantic")
class CityscapesSemantic(SemanticDataset):
    """The Cityscapes Semantic dataset consists of multiple sub-datasets
    related to semantic segmentation task.

    Attributes:
        root (str):
            The root directory that contains: train/val/test/... subdirectories.
        quality (str):
		    The quality of the semantic segmentation mask to use.
		    One of: [`gtFine`, `gtCoarse`]. Default: `gtFine`.
        split (str):
            The split to use.
            One of: ["train", "val", "test"].
        encoding (str):
            The format to use when creating the semantic segmentation mask.
            One of: [`id`, `trainId`, `catId`, `color`]. Default: `id`.
        extra (bool):
            Should use extra data? Those in the `train_extra` split are only available for `quality=gtCoarse`.
            Default: `False`.
        image_paths (list):
            A list of all image filepaths.
        semantic_paths (list):
            A list of all semantic segmentation image filepaths.
        label_paths (list):
            A list of all label filepaths that can provide extra info about the
            image and semantic segmentation image.
        data (list):
            The list of all `VisionData` objects.
        classlabels (ClassLabels, optional):
            The `ClassLabels` object contains all class-labels defined in the
            dataset.
        shape (tuple):
            The image shape as [H, W, C] Use to resize the image.
        encoding (str):
            The format to use when creating the semantic segmentation mask.
            One of: [`id`, `trainId`, `catId`, `color`, ...].
        label_formatter (object):
            The formatter in charge of formatting labels to the corresponding
            `label_format`.
        has_custom_labels (bool):
            Check if we have custom label files. If `True`, then those files
            will be loaded. Else, load the raw data from the dataset, convert
            them to our custom data format, and write to files.
        caching_labels (bool):
            Should overwrite the existing cached labels?
        caching_images (bool):
            Cache images into memory for faster training.
        write_labels (bool):
            After loading images and labels for the first time, we will convert
            it to our custom data format and write to files.
        fast_dev_run (bool):
            Take a small subset of the data for fast debug (i.e, like unit
            testing).
        augment (ImageAugment):
            A data object contains all hyperparameters for augmentation
            operations.
            > Note 1: The reason for this attributes is that sometimes we want
            to use custom augmentation operations that performs on both
            images and labels.
        transforms (callable, optional):
            A function/transform that takes input sample and its target as
            entry and returns a transformed version.
        transform (callable, optional):
            A function/transform that takes input sample as entry and returns
            a transformed version.
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        quality         : str                    = "gtFine",
        split           : str                    = "train",
        encoding        : str                    = "id",
        extra           : bool                   = False,
        shape           : Dim3                   = (720, 1280, 3),
        caching_labels  : bool                   = False,
        caching_images  : bool                   = False,
        write_labels    : bool                   = False,
        fast_dev_run    : bool                   = False,
        augment         : Union[str, Dict, None] = None,
        transforms      : Optional[Callable]     = None,
        transform       : Optional[Callable]     = None,
        target_transform: Optional[Callable]     = None,
        *args, **kwargs
    ):
        self.quality = quality
        self.extra   = extra
        assert quality  in ["gtFine", "gtCoarse"], \
            logger.error(f"Cityscapes Semantic dataset does not supports "
                         f"`quality`: `{quality}`.")
        assert encoding in ["id", "trainId", "catId", "color"], \
            logger.error(f"Cityscapes Semantic dataset does not supports "
                         f"`encoding`: `{encoding}`.")

        # NOTE: Load classlabels
        classlabel_path = os.path.join(root, quality, f"classlabels.json")
        if not os.path.isfile(classlabel_path):
            curr_dir        = os.path.dirname(os.path.abspath(__file__))
            classlabel_path = os.path.join(
                curr_dir, f"cityscapes_classlabels.json"
            )
        classlabels = ClassLabels.create_from_file(label_path=classlabel_path)
        
        super().__init__(
            root             = root,
            split            = split,
            classlabels      = classlabels,
            encoding         = encoding,
            shape            = shape,
            caching_labels   = caching_labels,
            caching_images   = caching_images,
            write_labels     = write_labels,
            fast_dev_run     = fast_dev_run,
            augment          = augment,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform,
            *args, **kwargs
        )
        
    # MARK: Pre-Load Data
    
    def list_files(self):
        """List image and label files.

        Todos:
        	- Look for image and label files in `split` directory.
        	- We should look for our custom label files first.
        	- If none is found, proceed to listing the images and raw label
        	  files.
        	- Also, load `classlabels`.
        	- After this method, these following attributes MUST be defined:
        	  `image_paths`, `semantic_paths`, `label_paths`,
        	  `has_custom_labels`, `classlabels`.
        """
        # NOTE: Flag some warnings
        if self.quality == "gtCoarse":
            logger.info(f"Cityscapes Semantic dataset only supports `split`: "
                        f"`train`, `train_extra`, or `val`. Get: {self.split}.")
        else:
            logger.info(f"Cityscapes Semantic dataset only supports `split`: "
                        f"`train`, `val`, or `test`. Get: {self.split}.")

        # NOTE: List image files
        image_patterns = [
            os.path.join(self.root, "leftImg8bit", self.split, "*", "*.png")
        ]
        if self.split == "train" and self.quality == "gtCoarse" and self.extra:
            image_patterns.append(
                os.path.join(
                    self.root, "leftImg8bit", "train_extra", "*", "*.png"
                )
            )

        image_paths = []
        for image_pattern in image_patterns:
            for image_path in glob.glob(image_pattern):
                image_paths.append(image_path)
        self.image_paths = unique(image_paths)  # Remove all duplicates files

        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices          = [random.randint(0, len(self.image_paths) - 1)
                                for _ in range(self.batch_size)]
            self.image_paths = [self.image_paths[i] for i in indices]
            
        # NOTE: List semantic files
        semantic_prefixes   = [path.replace("_leftImg8bit.png", "")
                               for path   in self.image_paths]
        semantic_prefixes   = [predix.replace("leftImg8bit", self.quality)
                               for predix in semantic_prefixes]
        self.semantic_paths = [f"{prefix}_{self.quality}_{self.encoding}.png"
                               for prefix in semantic_prefixes]

        # NOTE: List label files
        label_paths = [path.replace(self.quality, f"{self.quality}_customs")
                       for path in self.semantic_paths]
        label_paths = [path.replace(".png", ".json")
                       for path in label_paths]
        label_paths = [path for path in label_paths if os.path.isfile(path)]

        self.has_custom_labels = (
            (len(label_paths) == len(self.image_paths)) and
            (not self.caching_labels)
        )
        if not self.has_custom_labels:
            label_paths = [f"{prefix}_{self.quality}_polygons.json"
                           for prefix in semantic_prefixes]
        self.label_paths = label_paths
        
        # NOTE: Assertion
        assert len(self.image_paths) == len(self.semantic_paths), \
            (f"Number of images != Number of semantic images: "
             f"{len(self.image_paths)} != {len(self.semantic_paths)}.")
        logger.info(f"Number of images: {len(self.image_paths)}.")

    # MARK: Load Data
    
    def load_labels(
        self,
        image_path   : str,
        semantic_path: str,
        label_path   : Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                The image filepath.
            semantic_path (str):
                The semantic segmentation image filepath.
            label_path (str, optional):
                The label filepath. Default: `None`.
    
        Returns:
            data (VisionData):
                The `VisionData` object.
	    """
        # NOTE: If we have custom labels
        if label_path and os.path.isfile(label_path):
            if self.has_custom_labels or not self.caching_labels:
                return VisualDataHandler().load_from_file(
                    image_path    = image_path,
                    semantic_path = semantic_path,
                    label_path    = label_path
                )
            else:
                return CityscapesLabelHandler().load_from_file(
                    image_path  = image_path,
                    label_path  = label_path,
                    classlabels = self.classlabels
                )
        
        # NOTE: Parse info
        image_info = ImageInfo.from_file(image_path=image_path)
        
        if is_image_file(path=semantic_path):
            semantic_info = ImageInfo.from_file(image_path=semantic_path)
            return VisionData(image_info=image_info,
                              semantic_info=semantic_info)

        return VisionData(image_info=image_info)
    

# MARK: - CityscapesSemanticDataModule

@DATAMODULES.register(name="cityscapes")
@DATAMODULES.register(name="cityscapes_semantic")
class CityscapesSemanticDataModule(DataModule):
    """Cityscapes Semantic DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "cityscapes"),
        name       : str = "cityscapes_semantic",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_cfg = kwargs
        
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: Optional[Phase] = None):
        """There are also data operations you might want to perform on every
        GPU.

		Todos:
			- Count number of classes.
			- Build classlabels vocabulary.
			- Perform train/val/test splits.
			- Apply transforms (defined explicitly in your datamodule or
			  assigned in init).
			- Define collate_fn for you custom dataset.

		Args:
			phase (Phase, optional):
				The phase to use: [None, Phase.TRAINING, Phase.TESTING].
				Set to "None" to setup all train, val, and test data.
				Default: `None`.
        """
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, Phase.TRAINING]:
            self.train = CityscapesSemantic(
                root=self.dataset_dir, split="train", **self.dataset_cfg
            )
            self.val   = CityscapesSemantic(
                root=self.dataset_dir, split="val", **self.dataset_cfg
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = CityscapesSemantic(
                root=self.dataset_dir, split="test", **self.dataset_cfg
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()
        
    def load_classlabels(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        classlabel_path = os.path.join(
            current_dir, "cityscapes_classlabels.json"
        )
        self.classlabels = ClassLabels.create_from_file(
            label_path=classlabel_path
        )
    

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = data
    dm   = CityscapesSemanticDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize one sample
    data_iter                = iter(dm.val_dataloader)
    images, semantics, shape = next(data_iter)
    show_images(images=images,    nrow=2)
    show_images(images=semantics, nrow=2, figure_num=1)
    plt.show(block=True)
