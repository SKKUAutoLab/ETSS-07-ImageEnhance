#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Blur dataset and datamodule.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Callable
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
from torchkit.core.data.image_info import ImageInfo
from torchkit.core.data import VisionData
from torchkit.core.dataset import DataModule
from torchkit.core.dataset import EnhancementDataset
from torchkit.core.dataset import VisualDataHandler
from torchkit.core.fileio import is_json_file
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchkit.core.utils import Dim3

from torchkit.datasets.builder import DATAMODULES
from torchkit.datasets.builder import DATASETS
from torchkit.utils import datasets_dir

logger = logging.getLogger()

__all__ = ["Blur", "BlurDataModule"]


# MARK: - Data Config

data = {
    "name": "blur",
    # The datasets" name.
    "subset": ["all"],
    # The type of sub-dataset to use. Can also be a list to include multiple
    # subsets. One of: [`gopro`, `hide`, `realblur`, `all`, `None`]. When
    # `all` or `None`, all subsets will be included. Default: `all`.
    "shape": [512, 512, 3],
    # The image shape as [H, W, C]. This is compatible with OpenCV format.
    "batch_size": 4,
    # Number of samples in one forward & backward pass.
    "label_format": "custom",
    # The format to convert images and labels to when `get_item()`.
    # Each labels' format has each own directory:
    # annotations_<format>. Example: `.../train/annotations_yolo/...`
    # Supports:
    # - `custom`: uses our custom annotation format.
    # - `coco`  : uses Coco annotation format.
    # - `yolo`  : uses Yolo annotation format.
    # - `pascal`: uses Pascal annotation format.
    # Default: `yolo`.
    "caching_labels": True,
    # Should overwrite the existing cached labels? Default: `False`.
    "caching_images": False,
    # Cache images into memory for faster training. Default: `False`.
    "write_labels": False,
    # After loading images and labels for the first time, we will convert it
    # to our custom data format and write to files. If `True`, we will
    # overwrite these files. Default: `False`.
    "fast_dev_run": False,
    # Take a small subset of the data for fast debug (i.e, like unit testing).
    # Default: `False`.
    "shuffle": True,
    # Set to `True` to have the data reshuffled at every training epoch.
    # Default: `True`.
    "augment": {
        "hsv_h": 0.0,  # Image HSV-Hue augmentation (fraction).
        "hsv_s": 0.0,  # Image HSV-Saturation augmentation (fraction).
        "hsv_v": 0.0,  # Image HSV-Value augmentation (fraction).
        "rotate": 0.0,  # Image rotation (+/- deg).
        "translate": 0.0,  # Image translation (+/- fraction).
        "scale": 0.0,  # Image scale (+/- gain).
        "shear": 0.0,  # Image shear (+/- deg).
        "perspective": 0.0,  # Image perspective (+/- fraction), range 0-0.001.
        "flip_ud": 0.5,  # Image flip up-down (probability).
        "flip_lr": 0.5,  # Image flip left-right (probability).
        "mixup": 0.0,  # Image mixup (probability).
        "mosaic": False,  # Use mosaic augmentation.
        "rect": False,
        # Train model using rectangular images instead of square ones.
        "stride": 32,
        # When `rect_training=True`, reshape the image shapes as a multiply of
        # stride.
        "pad": 0.0,
        # When `rect_training=True`, pad the empty pixel with given values
    }
}


# MARK: - Blur

@DATASETS.register(name="blur")
class Blur(EnhancementDataset):
    """The Blur dataset consists of multiple datasets related to deblurring
    task.

    Attributes:
        root (str):
			The dataset root directory that contains: train/val/test/...
			subdirectories.
		subset (str, list, optional):
            The type of sub-dataset to use. Can also be a list to include
            multiple subsets. One of: [`gopro`, `hide`, `realblur`, `all`,
            `None`]. When `all` or `None`, all subsets will be included.
            Default: `all`.
		split (str):
			The split to use. One of: ["train", "val", "test"].
		image_paths (list):
			A list of all image filepaths.
		eimage_paths (list):
			A list of all enhanced image filepaths.
		label_paths (list):
			A list of all label filepaths that can provide extra info about the
			image and enhanced image.
		data (list):
			The list of all `VisionData` objects.
		classlabels (ClassLabels, optional):
			The `ClassLabels` object contains all class-labels defined in the
			dataset. Default: `None`.
		shape (tuple):
			The image shape as [H, W, C] Use to resize the image.
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
			Cache images into memory for faster training..
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
		collate_fn (callable, None):
			The collate function used to fused input items together when using
			`batch_size > 1`. This is used in the DataLoader wrapper.
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
        subset          : Union[str, list[str], Optional] = "all",
        split           : str                             = "train",
        shape           : Dim3                            = (720, 1280, 3),
        caching_labels  : bool                            = False,
        caching_images  : bool                            = False,
        write_labels    : bool                            = False,
        fast_dev_run    : bool                            = False,
        augment         : Union[str, dict, None]          = None,
        transforms      : Optional[Callable]              = None,
        transform       : Optional[Callable]              = None,
        target_transform: Optional[Callable]              = None,
        *args, **kwargs
    ):
        subset = [subset] if isinstance(subset, str) else subset
        if "all" in subset or subset is None:
            subset = ["gopro", "hide", "realblur"]
        self.subset = subset

        super().__init__(
            root             = root,
            split            = split,
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
        	  `image_files`, `eimage_files`, `label_files`,
        	  `has_custom_labels`, `classlabels`.
        """
        # NOTE: List all files
        if "gopro" in self.subset:
            self.list_gopro_files()
        if "hide" in self.subset:
            self.list_hide_files()
        if "realblur" in self.subset:
            self.list_realblur_files()
            
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths  = [self.image_paths[i]  for i in indices]
            self.eimage_paths = [self.eimage_paths[i] for i in indices]
            self.label_paths  = [self.label_paths[i]  for i in indices]
        
        # NOTE: Assertion
        assert len(self.image_paths) == len(self.eimage_paths), \
            (f"Number of images != Number of enhanced images: "
             f"{len(self.image_paths)} != {len(self.eimage_paths)}.")
        logger.info(f"Number of images: {len(self.image_paths)}.")
        
    def list_gopro_files(self):
        """List all GoPro files.
        """
        logger.info(f"GoPro dataset only supports `split`: `train`. "
                    f"Get: {self.split}.")
        eimage_pattern = os.path.join(
            self.root, "gopro", self.split, "*", "sharp", "*.png"
        )
        for eimage_path in glob.glob(eimage_pattern):
            image_path = eimage_path.replace("sharp", "blur")
            label_path = eimage_path.replace("sharp", "customs")
            label_path = label_path.replace(".png", ".json")
            self.image_paths.append(image_path)
            self.eimage_paths.append(eimage_path)
            self.label_paths.append(label_path)
            self.has_custom_labels = is_json_file(path=label_path)
    
    def list_hide_files(self):
        """List all HIDE data.
        """
        logger.info(f"HIDE dataset only supports `split`: `train` or `test`. "
                    f"Get: {self.split}.")
        eimage_pattern = os.path.join(
            self.root, "hide", self.split, "sharp", "*.png"
        )
        for eimage_path in glob.glob(eimage_pattern):
            image_path = eimage_path.replace("sharp", "blur")
            label_path = eimage_path.replace("sharp", "customs")
            label_path = label_path.replace(".jpg", ".json")
            self.image_paths.append(image_path)
            self.eimage_paths.append(eimage_path)
            self.label_paths.append(label_path)
            self.has_custom_labels = is_json_file(path=label_path)
    
    def list_realblur_files(self):
        """List all RealBlur data.
        """
        logger.info(f"RealBlur dataset only supports `split`: `train` or "
                    f"`test`. Get: {self.split}.")
        eimage_pattern = os.path.join(
            self.root, "realblur", self.split, "j", "*", "gt", "*.png"
        )
        for eimage_path in glob.glob(eimage_pattern):
            image_path = eimage_path.replace("gt", "blur")
            label_path = eimage_path.replace("gt", "customs")
            label_path = label_path.replace(".jpg", ".json")
            self.image_paths.append(image_path)
            self.eimage_paths.append(eimage_path)
            self.label_paths.append(label_path)
            self.has_custom_labels = is_json_file(path=label_path)

    # MARK: Load Data
    
    def load_labels(
        self,
        image_path : str,
        eimage_path: str,
        label_path : Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

		Args:
			image_path (str):
				The image file.
			eimage_path (str):
				The enhanced image file.
			label_path (str, optional):
				The label file. Default: `None`.
	
		Returns:
			data (VisionData):
				The `VisionData` object.
		"""
        # NOTE: If we have custom labels
        if self.has_custom_labels and label_path:
            return VisualDataHandler().load_from_file(
                image_path  = image_path,
                label_path  = label_path,
                eimage_path = eimage_path
            )

        # NOTE: Parse info
        image_info  = ImageInfo.from_file(image_path=image_path)
        eimage_info = ImageInfo.from_file(image_path=eimage_path)
        
        return VisionData(image_info=image_info, eimage_info=eimage_info)


# MARK: - BlurDataModule

@DATAMODULES.register(name="blur")
class BlurDataModule(DataModule):
    """Rain DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "blur"),
        name       : str = "blur",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_kwargs = kwargs
        
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
            self.train = Blur(root=self.dataset_dir, split="train",
                              **self.dataset_kwargs)
            self.val   = Blur(root=self.dataset_dir, split="val",
                              **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = Blur(root=self.dataset_dir, split="test",
                             **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()
            
    def load_classlabels(self):
        pass
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = data
    dm   = BlurDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize one sample
    data_iter              = iter(dm.train_dataloader)
    images, eimages, shape = next(data_iter)
    show_images(images=images,  nrow=2)
    show_images(images=eimages, nrow=2, figure_num=1)
    plt.show(block=True)
