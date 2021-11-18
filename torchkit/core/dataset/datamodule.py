#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all datamodules present in the `torchkit.datasets` packages.
"""

from __future__ import annotations

import logging
from typing import Callable
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchkit.core.runner import Phase
from torchkit.core.utils import EvalDataLoaders
from torchkit.core.utils import TrainDataLoaders

logger = logging.getLogger()


# MARK: - DataModule

class DataModule(pl.LightningDataModule):
    """Base class for all data module present in the `torchkit.datasets`
    packages.
    
    Attributes:
        dataset_dir (str):
            The path to the dataset directory.
        name (str):
            The dataset name.
        shape (tuple):
            The image shape as [H, W, C]. This is compatible with OpenCV format.
        batch_size (int):
            Number of training samples in one forward & backward pass.
        shuffle (bool):
             If `True`, reshuffle the data at every training epoch.
        train (Dataset):
            The train dataset.
        val (Dataset):
            The val dataset.
        test (Dataset):
            The test dataset.
        predict (Dataset):
            The predict dataset.
        classlabels (ClassLabels, optional):
            The `ClassLabels` object contains all class-labels defined in the
            dataset.
        collate_fn (callable, optional):
            The collate function used to fused input items together when using
            `batch_size > 1`.
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
        dataset_dir     : str,
        name            : str,
        shape           : tuple,
        batch_size      : int                = 1,
        shuffle         : bool               = True,
        collate_fn      : Optional[Callable] = None,
        transforms      : Optional[Callable] = None,
        transform       : Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dataset_dir      = dataset_dir
        self.name             = name
        self.shape            = shape
        self.batch_size       = batch_size
        self.shuffle          = shuffle
        self.train            = None
        self.val              = None
        self.test             = None
        self.predict          = None
        self.classlabels      = None
        self.collate_fn       = collate_fn
        self.transforms       = transforms
        self.transform        = transform
        self.target_transform = target_transform

    # MARK: Property
   
    @property
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        if self.classlabels is not None:
            return self.classlabels.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """Return number of workers used in the data loading pipeline."""
        # NOTE: Set `num_workers` = the number of cpus on the current machine
        # to avoid bottleneck
        return 0  # os.cpu_count() - 1

    @property
    def train_dataloader(self) -> Optional[TrainDataLoaders]:
        """Implement one or more PyTorch DataLoaders for training."""
        if self.train:
            return DataLoader(
                dataset     = self.train,
                batch_size  = self.batch_size,
                shuffle     = self.shuffle,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
                collate_fn  = self.collate_fn
            )
        return None

    @property
    def val_dataloader(self) -> Optional[EvalDataLoaders]:
        """Implement one or more PyTorch DataLoaders for validation."""
        if self.val:
            return DataLoader(
                dataset     = self.val,
                batch_size  = self.batch_size,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
                collate_fn  = self.collate_fn
            )
        return None

    @property
    def test_dataloader(self) -> Optional[EvalDataLoaders]:
        """Implement one or more PyTorch DataLoaders for testing."""
        if self.test:
            return DataLoader(
                dataset     = self.test,
                batch_size  = self.batch_size,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
                collate_fn  = self.collate_fn
            )
        return None
    
    def predict_dataloader(self) -> Optional[EvalDataLoaders]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        if self.predict:
            return DataLoader(
                dataset     = self.predict,
                batch_size  = self.batch_size,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
                collate_fn  = self.collate_fn
            )
        return None
    
    # MARK: Prepare Data

    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        raise NotImplementedError

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
                The stage to use: [None, Phase.TRAINING, Phase.TESTING].
                Set to `None` to setup all train, val, and test data.
        """
        raise NotImplementedError
