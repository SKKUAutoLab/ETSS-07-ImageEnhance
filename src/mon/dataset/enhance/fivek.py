#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIT-Adobe FiveK Datasets."""

from __future__ import annotations

__all__ = [
    "FiveKA",
    "FiveKADataModule",
    "FiveKB",
    "FiveKBDataModule",
    "FiveKC",
    "FiveKCDataModule",
    "FiveKD",
    "FiveKDDataModule",
    "FiveKE",
    "FiveKEDataModule",
    "FiveKInit",
    "FiveKInitDataModule",
]

from collections import defaultdict
from typing import Literal

import numpy as np
import torch

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="fivek_init")
class FiveKInit(MultimodalDataset):
    """A special MIT-Adobe FiveK dataset used for initiliazing the model.
    """
    
    tasks : list[Task]  = [Task.LLIE, Task.RETOUCH]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image_ex": ImageAnnotation,
        "image_bc": ImageAnnotation,
        "image_vb": ImageAnnotation,
        "ref_ex"  : ImageAnnotation,
        "ref_bc"  : ImageAnnotation,
        "ref_vb"  : ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        self.file_ex   = defaultdict(list)
        self.file_bc   = defaultdict(list)
        self.file_vb   = defaultdict(list)
        self.file_keys = []
        super().__init__(root=root, *args, **kwargs)
     
    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary containing the datapoint and metadata at the
        given :obj:`index`.
        """
        key = self.file_keys[index]
        A_ex, B_ex = np.random.choice(self.file_ex[key], 2, replace=False)
        A_bc, B_bc = np.random.choice(self.file_bc[key], 2, replace=False)
        A_vb, B_vb = np.random.choice(self.file_vb[key], 2, replace=False)
        
        val_ex     = torch.tensor((int(B_ex.stem.split("-")[-1]) - int(A_ex.stem.split("-")[-1])) / 20).float()
        val_bc     = torch.tensor((int(B_bc.stem.split("-")[-1]) - int(A_bc.stem.split("-")[-1])) / 20).float()
        val_vb     = torch.tensor((int(B_vb.stem.split("-")[-1]) - int(A_vb.stem.split("-")[-1])) / 20).float()
        
        image_ex   = ImageAnnotation(path=A_ex, root=self.root)
        ref_ex     = ImageAnnotation(path=B_ex, root=self.root)
        image_bc   = ImageAnnotation(path=A_bc, root=self.root)
        ref_bc     = ImageAnnotation(path=B_bc, root=self.root)
        image_vb   = ImageAnnotation(path=A_vb, root=self.root)
        ref_vb     = ImageAnnotation(path=B_vb, root=self.root)
        
        datapoint = {
            "image_ex": image_ex.data,
            "image_bc": image_bc.data,
            "image_vb": image_vb.data,
            "ref_ex"  : ref_ex.data,
            "ref_bc"  : ref_bc.data,
            "ref_vb"  : ref_vb.data,
        }
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, keepdim=False, normalize=True)
        datapoint |= {
            "val_ex"       : val_ex,
            "val_bc"       : val_bc,
            "val_vb"       : val_vb,
            "image_ex_meta": image_ex.meta,
            "image_bc_meta": image_bc.meta,
            "image_vb_meta": image_vb.meta,
            "ref_ex_meta"  : ref_ex.meta,
            "ref_bc_meta"  : ref_bc.meta,
            "ref_vb_meta"  : ref_vb.meta,
        }
        
        # Return
        return datapoint
    
    def __len__(self) -> int:
        return len(self.file_keys)
       
    def get_data(self):
        patterns = [
            self.root / "fivek_init",
        ]
        
        # Exposure images
        file_ex = defaultdict(list)
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("exposure/*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} exposure images"
                ):
                    if path.is_image_file():
                        idx_ex = path.stem.split("-")[0]
                        file_ex[idx_ex].append(path)
        
        # Black clipping images
        file_bc = defaultdict(list)
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("black_clipping/*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} black clipping images"
                ):
                    if path.is_image_file():
                        idx_bc = path.stem.split("-")[0]
                        file_bc[idx_bc].append(path)
        
        # Vibrance images
        file_vb = defaultdict(list)
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("vibrance/*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} vibrance images"
                ):
                    if path.is_image_file():
                        idx_vb = path.stem.split("-")[0]
                        file_vb[idx_vb].append(path)
        
        self.file_ex   = file_ex
        self.file_bc   = file_bc
        self.file_vb   = file_vb
        self.file_keys = list(self.file_ex.keys())
    
    def verify_data(self):
        """Verify dataset."""
        pass
    

@DATASETS.register(name="fivek_a")
class FiveKA(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert A ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_a" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
    
        self.datapoints["image"] = images


@DATASETS.register(name="fivek_b")
class FiveKB(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert B ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_b" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="fivek_c")
class FiveKC(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert C ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_c" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="fivek_d")
class FiveKD(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert D ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_d" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} lq images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="fivek_e")
class FiveKE(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert E ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_e" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} lq images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATAMODULES.register(name="fivek_init")
class FiveKInitDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKInit(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = None
        if stage in [None, "test"]:
            self.test  = None
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_a")
class FiveKADataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKA(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKA(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKA(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_b")
class FiveKBDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKB(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKB(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKB(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_c")
class FiveKCDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKC(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKC(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKC(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_d")
class FiveKDDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKD(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKD(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKD(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_e")
class FiveKEDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = FiveKE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKE(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKE(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
