#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for all datasets."""

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "DatapointAttributes",
    "Dataset",
    "IterableDataset",
    "Subset",
    "TensorDataset",
    "random_split",
]

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import (
    ChainDataset, ConcatDataset, IterableDataset, random_split, Subset, TensorDataset,
)

from mon.constants import Split, Task
from mon.core import pathlib, rich
from mon.core.types.annotations import Annotation, ClassLabels


# ----- Datapoint -----
class DatapointAttributes(dict[str, Annotation]):
    """Holds datapoint attributes as a ``dict``.

    Args:
        args: Positional arguments for ``dict`` initialization.
        kwargs: Keyword arguments for ``dict`` initialization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

# ----- Dataset -----
class Dataset(dataset.Dataset, ABC):
    """Base class for all datasets.

    Attributes:
        tasks: List of supported tasks.
        splits: List of supported splits.
        has_test_annotations: If ``True``, test set has labels. Default is ``False``.
        datapoint_attrs: Dict of datapoint attributes (keys: names, values: types).
        classlabels: ``ClassLabels`` with supported labels. Default is ``None``.

    Args:
        root: Root dir with split subdirs. Default is ``None``.
        split: Data split to use. Default is ``Split.TRAIN``.
        transform: Transformations for input/target. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``False``.
    """
    
    tasks : list[Task]  = []
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    datapoint_attrs     = DatapointAttributes({})
    has_test_annotations: bool = False
    classlabels: ClassLabels   = None
    
    def __init__(
        self,
        root      : pathlib.Path,
        split     : Split = Split.TRAIN,
        transform : Any   = None,
        to_tensor : bool  = False,
        cache_data: bool  = False,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root       = pathlib.Path(root)
        self.split      = split
        self.transform  = None
        self.to_tensor  = to_tensor
        self.verbose    = verbose
        self.index      = 0  # Used with `__iter__` and `__next__`
        self.datapoints = {}
        self.init_transform(transform)
        self.init_datapoints()
        self.init_data(cache_data=cache_data)
        
    # ----- Magic Methods -----
    def __del__(self):
        """Closes the dataset."""
        self.close()
    
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at given ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            ``dict`` with datapoint and metadata.
        """
        pass
    
    def __iter__(self):
        """Gets total number of datapoints.

        Returns:
            Number of datapoints in dataset.
        """
        self.reset()
        return self
    
    @abstractmethod
    def __len__(self) -> int:
        """Gets the total number of datapoints.

        Returns:
            Number of datapoints in the dataset.
        """
        pass
    
    def __next__(self) -> dict:
        """Gets the next datapoint and metadata.

        Returns:
            Dict with next datapoint and metadata.

        Raises:
            StopIteration: If index exceeds dataset length.
        """
        if self.index >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.index)
        self.index += 1
        return result
    
    def __repr__(self) -> str:
        """Returns string representation of dataset.

        Returns:
            Formatted string with dataset details.
        """
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]
        return "\n".join(lines)
    
    # ----- Properties -----
    @property
    def disable_pbar(self) -> bool:
        """Indicates if progress bar is disabled.

        Returns:
            ``True`` if progress bar disabled, ``False`` otherwise.
        """
        return not self.verbose
    
    @property
    def has_annotations(self) -> bool:
        """Checks if images have annotations.

        Returns:
            ``True`` if annotations exist, ``False`` otherwise.
        """
        return (
            (
                self.has_test_annotations
                and self.split in [Split.TEST, Split.PREDICT]
            )
            or (self.split in [Split.TRAIN, Split.VAL])
        )
    
    @property
    def hash(self) -> int:
        """Gets total hash value of all files.

        Returns:
            Integer sum of file hash values in bytes.
        """
        sum = 0
        for k, v in self.datapoints.items():
            if isinstance(v, list):
                for a in v:
                    if a and hasattr(a, "meta"):
                        sum += a.meta.get("hash", 0)
        return sum
    
    @property
    def main_attribute(self) -> str:
        """Gets the main dataset attribute.

        Returns:
            First key from ``datapoint_attrs`` as string.
        """
        return next(iter(self.datapoint_attrs.keys()))
    
    @property
    def new_datapoint(self) -> dict:
        """Creates a new datapoint with default values.

        Returns:
            Dict with attribute keys set to ``None``.
        """
        return {k: None for k in self.datapoints.keys()}
    
    @property
    def split(self) -> Split:
        """Gets the current dataset split.

        Returns:
            Current ``Split`` value.
        """
        return self._split
    
    @split.setter
    def split(self, split: Split):
        """Sets the dataset split.

        Args:
            split: Split value to set.

        Raises:
            ValueError: If ``split`` not in supported splits.
        """
        split = Split[split] if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(f"[split] must be one of {self.splits}, got {split}.")
    
    @property
    def split_str(self) -> str:
        """Gets string representation of the split.

        Returns:
            String value of current split.
        """
        return self.split.value
    
    # ----- Initialize -----
    @abstractmethod
    def init_transform(self, transform: Any = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        pass

    def init_datapoints(self):
        """Initializes the datapoints dictionary.

        Raises:
            ValueError: If ``datapoint_attrs`` has no attributes.
        """
        if not self.datapoint_attrs:
            raise ValueError("[datapoint_attrs] has no defined attributes.")
        self.datapoints = {k: list[v]() for k, v in self.datapoint_attrs.items()}
    
    def init_data(self, cache_data: bool = False):
        """Initializes dataset data.

        Args:
            cache_data: If ``True``, caches data to disk. Default is ``False``.
        """
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.list_data()
        
        self.filter_data()
        self.verify_data()
        
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            pathlib.delete_cache(cache_file)
    
    @abstractmethod
    def list_data(self):
        """Lists all data files in the dataset."""
        pass
    
    def cache_data(self, path: pathlib.Path):
        """Caches data to the specified path.

        Args:
            path: Path to save the cache.
        """
        hash_ = 0
        if path.is_cache_file():
            cache = torch.load(path)
            hash_ = cache.get("hash", 0)
        
        if self.hash != hash_:
            cache = self.datapoints | {"hash": self.hash}
            torch.save(cache, str(path))
            if self.verbose:
                rich.console.log(f"Cached data to: {path}")
    
    def load_cache(self, path: pathlib.Path):
        """Loads cached data from specified path.

        Args:
            path: Path to load cache from.
        """
        self.datapoints = torch.load(path)
        self.datapoints.pop("hash", None)
    
    @abstractmethod
    def filter_data(self):
        """Filters unwanted datapoints."""
        pass
    
    @abstractmethod
    def verify_data(self):
        """Verifies the dataset."""
        pass
    
    @abstractmethod
    def reset(self):
        """Resets the dataset."""
        pass
    
    @abstractmethod
    def close(self):
        """Closes and releases the dataset."""
        pass
    
    # ----- Data Retrieval -----
    @abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict containing the datapoint.
        """
        pass
    
    @abstractmethod
    def get_meta(self, index: int) -> dict:
        """Gets metadata at specified index.

        Args:
            index: Index of metadata.

        Returns:
            Dict containing the metadata.
        """
        pass
    
    @classmethod
    def collate_fn(cls, batch: list[dict]) -> dict:
        """Collates input items for batch processing.

        Args:
            batch: List of dicts from dataset.

        Returns:
            Collated ``dict`` for ``torch.utils.data.DataLoader``.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        for k, v in zipped.items():
            if k not in cls.datapoint_attrs:
                continue
            collate_fn = getattr(cls.datapoint_attrs[k], "collate_fn", None)
            if collate_fn and v is not None:
                zipped[k] = collate_fn(batch=v)
        return zipped
