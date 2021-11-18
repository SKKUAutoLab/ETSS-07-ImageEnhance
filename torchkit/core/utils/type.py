#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom data types.
"""

from __future__ import annotations

import functools
import logging
import types
from typing import Any
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger()


# MARK: - Templates

# Template for arguments which can be supplied as a tuple, or which can be a
# scalar which PyTorch will internally broadcast to a tuple. Comes in several
# variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d
# operations.
T                      = TypeVar("T")
ScalarOrTupleAnyT      = Union[T, tuple[T, ...]]
ScalarOrTuple1T        = Union[T, tuple[T]]
ScalarOrTuple2T        = Union[T, tuple[T, T]]
ScalarOrTuple3T        = Union[T, tuple[T, T, T]]
ScalarOrTuple4T        = Union[T, tuple[T, T, T, T]]
ScalarOrTuple5T        = Union[T, tuple[T, T, T, T, T]]
ScalarOrTuple6T        = Union[T, tuple[T, T, T, T, T, T]]
ScalarListOrTupleAnyT  = Union[T, list[T], tuple[T, ...]]
ScalarOrCollectionAnyT = Union[T, list[T], tuple[T, ...], dict[Any, T]]
ListOrTupleAnyT        = Union[   list[T], tuple[T, ...]]


# MARK: - Object and Function

FuncCls = Union[type, str, types.FunctionType, functools.partial]


# MARK: - Custom Data Types

ID         = Union[int, str]
Indexes    = ScalarListOrTupleAnyT[int]
Color      = Union[list[int, int, int], tuple[int, int, int]]
LabelTypes = ScalarListOrTupleAnyT[str]
Tasks      = ScalarListOrTupleAnyT[str]

Tensors    = ScalarOrCollectionAnyT[torch.Tensor]
Arrays     = ScalarOrCollectionAnyT[np.ndarray]
Metrics    = Union[dict[str, torch.Tensor], dict[str, np.ndarray]]

Image      = Union[torch.Tensor, np.ndarray]
ImageList  = Union[list[torch.Tensor],       list[np.ndarray]]
ImageTuple = Union[tuple[torch.Tensor, ...], tuple[np.ndarray, ...]]
ImageDict  = Union[dict[str, torch.Tensor],  dict[str, np.ndarray]]
Images     = Union[Tensors, Arrays]


# MARK: - Size Parameters

Dim2     = tuple[int, int]
Dim3     = tuple[int, int, int]

SizeAnyT = ScalarOrTupleAnyT[int]
Size1T   = ScalarOrTuple1T[int]
Size2T   = ScalarOrTuple2T[int]
Size3T   = ScalarOrTuple3T[int]
Size4T   = ScalarOrTuple4T[int]
Size5T   = ScalarOrTuple5T[int]
Size6T   = ScalarOrTuple6T[int]


# MARK: - DataLoader

TrainDataLoaders = Union[
    DataLoader,
    Sequence[DataLoader],
    Sequence[Sequence[DataLoader]],
    Sequence[dict[str, DataLoader]],
    dict[str, DataLoader],
    dict[str, dict[str, DataLoader]],
    dict[str, Sequence[DataLoader]],
]
EvalDataLoaders = Union[DataLoader, Sequence[DataLoader]]


# MARK: - Training Parameters

ForwardXYOutput = tuple[Tensors, Optional[Metrics]]
ForwardOutput   = Union[Tensors, ForwardXYOutput]
