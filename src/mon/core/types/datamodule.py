#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class for all datamodules."""

__all__ = [
    "DataModule",
]

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import lightning
import torch
from torch.utils import data

from mon.constants import Task
from mon.core import rich, type_extensions, device as D
from mon.core.types import dataset


# ----- DataModule -----
class DataModule(lightning.LightningDataModule, ABC):
    """Base class for all datamodules.

    Attributes:
        dataset_kwargs: Dict of default args for datasets, e.g.,
            train = Dataset(split='train', **self.dataset_kwargs).

    Args:
        datasets: Dataset(s) to use.
        batch_size: Samples per forward pass. Default is ``1``.
        devices: Devices to use. Default is ``cpu``.
        shuffle: If ``True``, reshuffles data each epoch. Default is ``True``.
        collate_fn: Function to fuse datapoints for ``batch_size`` > ``1``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    tasks: list[Task] = []
    
    def __init__(
        self,
        datasets   : Any      = None,
        batch_size : int      = 1,
        devices    : int | str | torch.device = "cpu",
        shuffle    : bool     = True,
        collate_fn : Callable = None,
        verbose    : bool     = True,
        *args, **kwargs
    ):
        super().__init__()
        self.batch_size     = batch_size
        # self.devices        = type_extensions.to_list(devices)
        self.devices        = D.set_device(devices)
        self.shuffle        = shuffle
        self.collate_fn     = collate_fn
        self.verbose        = verbose
        self.dataset_kwargs = kwargs | {"verbose": verbose}

        if isinstance(datasets, dict):
            self.train   = datasets.pop("train")   if "train"   in datasets else None
            self.val     = datasets.pop("val")     if "val"     in datasets else None
            self.test    = datasets.pop("test")    if "test"    in datasets else None
            self.predict = datasets.pop("predict") if "predict" in datasets else None
            self.dataset_kwargs.update(datasets)
        elif isinstance(datasets, dataset.Dataset):
            self.train = self.val = self.test = self.predict = datasets
        else:
            self.train = self.val = self.test = self.predict = None
        
        self.classlabels = None
    
    # ----- Properties -----
    @property
    def num_workers(self) -> int:
        """Gets the number of workers for data loading.

        Returns:
            Number of workers (4 times device count).
        """
        devices = self.devices if isinstance(self.devices, (list, tuple)) else [self.devices]
        return 4 * len(devices)
    
    @property
    def train_dataloader(self) -> data.DataLoader | None:
        """Gets a DataLoader for the train dataset.

        Returns:
            ``DataLoader`` for train data or ``None`` if unavailable.
        """
        if not self.train:
            return None
        
        self.classlabels = getattr(self.train, "classlabels", self.classlabels)
        return data.DataLoader(
            dataset            = self.train,
            batch_size         = self.batch_size,
            shuffle            = self.shuffle,
            num_workers        = self.num_workers,
            pin_memory         = True,
            drop_last          = False,
            collate_fn         = getattr(self.train, "collate_fn", self.collate_fn),
            generator          = torch.Generator(device=self.devices),
            persistent_workers = True
        )
    
    @property
    def val_dataloader(self) -> data.DataLoader | None:
        """Gets a DataLoader for the val dataset.

        Returns:
            ``DataLoader`` for val data or ``None`` if unavailable.
        """
        if not self.val:
            return None
        
        return data.DataLoader(
            dataset            = self.val,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            drop_last          = False,
            collate_fn         = getattr(self.val, "collate_fn", self.collate_fn),
            generator          = torch.Generator(device=self.devices),
            persistent_workers = True
        )
    
    @property
    def test_dataloader(self) -> data.DataLoader | None:
        """Gets a DataLoader for the test dataset.
    
        Returns:
            ``DataLoader`` for test data or ``None`` if unavailable.
        """
        if not self.test:
            return None
        
        return data.DataLoader(
            dataset            = self.test,
            batch_size         = 1,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            drop_last          = False,
            collate_fn         = getattr(self.test, "collate_fn", self.collate_fn),
            generator          = torch.Generator(device=self.devices),
            persistent_workers = True
        )
    
    @property
    def predict_dataloader(self) -> data.DataLoader | None:
        """Gets a DataLoader for the predict dataset.
    
        Returns:
            ``DataLoader`` for predict data or ``None`` if unavailable.
        """
        if not self.predict:
            return None
        
        return data.DataLoader(
            dataset            = self.predict,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            drop_last          = True,
            collate_fn         = self.collate_fn,
            generator          = torch.Generator(device=self.devices),
            persistent_workers = True
        )
    
    @property
    def can_log(self) -> bool:
        """Checks if logging is enabled.

        Returns:
            ``True`` if logging enabled, ``False`` otherwise.
        """
        return self.verbose and (self.trainer is None or self.trainer.global_rank == 0)
    
    # ----- Initialize -----
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """Prepares data for disk or single-GPU tasks.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        pass
    
    @abstractmethod
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up data for every device.

        Args:
            stage: Running stage (``train``, ``test``, ``predict``, or ``None``).
        """
        pass
    
    def get_classlabels(self):
        """Loads class labels from datasets."""
        for dataset in [self.train, self.val, self.test, self.predict]:
            if dataset:
                self.classlabels = getattr(dataset, "classlabels", None)
                return
        rich.console.log("[yellow]No classlabels found")
    
    def split_train_val(
        self,
        dataset    : dataset.Dataset,
        split_ratio: float = 0.8,
        full_train : bool  = True
    ):
        """Splits dataset into train and val sets.

        Args:
            dataset: Dataset to split.
            split_ratio: Train set ratio. Default is ``0.8``.
            full_train: If ``True``, uses full dataset for train. Default is ``True``.
        """
        train_size       = int(split_ratio * len(dataset))
        val_size         = len(dataset) - train_size
        train, self.val  = data.random_split(dataset, [train_size, val_size])
        self.train       = dataset if full_train else train
        self.classlabels = getattr(dataset, "classlabels", None)
        self.collate_fn  = getattr(dataset, "collate_fn",  None)
    
    def summarize(self):
        """Prints a dataset summary."""
        table = rich.table.Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        table.add_row("1", "train",       f"{len(self.train) if self.train else None}")
        table.add_row("2", "val",         f"{len(self.val)   if self.val   else None}")
        table.add_row("3", "test",        f"{len(self.test)  if self.test  else None}")
        table.add_row("4", "classlabels", f"{self.classlabels.num_classes if self.classlabels else None}")
        table.add_row("5", "batch_size",  f"{self.batch_size}")
        table.add_row("6", "num_workers", f"{self.num_workers}")
        rich.console.log(table)
        
        if self.classlabels:
            self.classlabels.print()
