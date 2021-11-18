#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations on containers.
"""

from __future__ import annotations

import collections
import itertools
import logging
from collections import abc
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
from multipledispatch import dispatch

from .type import Tensors, Arrays

logger = logging.getLogger()


# MARK: - Validate

def is_seq_of(
	seq          : Sequence,
	expected_type: type,
	seq_type     : Optional[type] = None
) -> bool:
	"""Check whether it is a sequence of some type.

	Args:
		seq (Sequence):
			The sequence to be checked.
		expected_type (type):
			Expected type of sequence items.
		seq_type (type, optional):
			Expected sequence type.
	"""
	if seq_type is None:
		exp_seq_type = abc.Sequence
	else:
		assert isinstance(seq_type, type)
		exp_seq_type = seq_type
	if not isinstance(seq, exp_seq_type):
		return False
	for item in seq:
		if not isinstance(item, expected_type):
			return False
	return True


def is_list_of(seq: list, expected_type: type) -> bool:
	"""Check whether it is a list of some type. A partial method of
	`is_seq_of()`.
	"""
	return is_seq_of(seq=seq, expected_type=expected_type, seq_type=list)


def is_tuple_of(seq: tuple, expected_type: type) -> bool:
	"""Check whether it is a tuple of some type. A partial method of
	`is_seq_of()`."""
	return is_seq_of(seq=seq, expected_type=expected_type, seq_type=tuple)


def is_dict_of(d: dict, expected_type: type) -> bool:
	"""Check whether it is a dict of some type."""
	assert isinstance(expected_type, type)
	return all(isinstance(v, expected_type) for k, v in d.items())


# MARK: - Cast

def to_iter(
	inputs     : Iterable,
	dst_type   : type,
	return_type: Optional[type] = None
):
	"""Cast elements of an iterable object into some type.
	
	Args:
		inputs (Iterable):
			The input object.
		dst_type (type):
			Destination type.
		return_type (type, optional):
			If specified, the output object will be converted to this type,
			otherwise an iterator.
	"""
	if not isinstance(inputs, abc.Iterable):
		raise TypeError("`inputs` must be an iterable object.")
	if not isinstance(dst_type, type):
		raise TypeError("`dst_type` must be a valid type.")

	out_iterable = map(dst_type, inputs)

	if return_type is None:
		return out_iterable
	else:
		return return_type(out_iterable)


def to_list(inputs: Iterable, dst_type: type):
	"""Cast elements of an iterable object into a list of some type. A partial
	method of `to_iter()`.
	"""
	return to_iter(inputs=inputs, dst_type=dst_type, return_type=list)


def to_tuple(inputs: Iterable, dst_type: type):
	"""Cast elements of an iterable object into a tuple of some type. A partial
	method of `to_iter()`."""
	return to_iter(inputs=inputs, dst_type=dst_type, return_type=tuple)


def to_ntuple(n: int):
	"""A helper functions to cast input to n-tuple."""
	def parse(x) -> tuple:
		if isinstance(x, collections.abc.Iterable):
			return tuple(x)
		return tuple(itertools.repeat(x, n))
	return parse


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)


def to_4d_tensor(x: Union[Tensors, Arrays]) -> torch.Tensor:
	"""Convert to a 4D-tensor. The output will be:
		- Single 3D-tensor will be expanded to a single 4D-tensor.
		- Single 4D-tensor will remain the same.
		- Sequence of 3D-tensors will be stacked into a 4D-tensor.
		- Sequence of 4D-tensors will remain the same.
	"""
	if isinstance(x, torch.Tensor):
		if x.dim() == 3:
			x = x.unsqueeze(dim=0)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, np.ndarray):
		x = torch.from_numpy(x)
		return to_4d_tensor(x=x)
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		if all(3 <= x_.ndim < 4 for x_ in x):
			return to_4d_tensor(x=torch.stack(x, dim=0))
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		x = [torch.from_numpy(_x) for _x in x]
		return to_4d_tensor(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_4d_tensor(x=np.array(x))
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_4d_array(x: Union[Tensors, Arrays]) -> np.ndarray:
	"""Convert to a 4D-array. The output will be:
		- Single 3D-array will be expanded to a single 4D-array.
		- Single 4D-array will remain the same.
		- Sequence of 3D-arrays will be stacked into a 4D-array.
		- Sequence of 4D-arrays will remain the same.
	"""
	if isinstance(x, np.ndarray):
		if x.ndim == 3:
			x = np.expand_dims(x, axis=0)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
		return to_4d_array(x=x)
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		if all(3 <= x_.ndim < 4 for x_ in x):
			return to_4d_array(x=np.stack(x))
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
	
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		x = [_x.detach().cpu().numpy() for _x in x]
		return to_4d_array(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_4d_array(x=x)
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_5d_tensor(x: Union[Tensors, Arrays]) -> torch.Tensor:
	"""Convert to a 5D-tensor."""
	if isinstance(x, torch.Tensor):
		if x.dim() == 3:
			x = x.unsqueeze(dim=0)
			x = x.unsqueeze(dim=0)
		elif x.dim() == 4:
			x = x.unsqueeze(dim=0)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, np.ndarray):
		return to_5d_tensor(x=torch.from_numpy(x))
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		if all(3 <= x_.ndim < 4 for x_ in x):
			return to_5d_tensor(x=torch.stack(x, dim=0))
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		x = [torch.from_numpy(_x) for _x in x]
		return to_5d_tensor(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_5d_tensor(x=x)
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_5d_array(x: Union[Tensors, Arrays]) -> np.ndarray:
	"""Convert to a 5D-array."""
	if isinstance(x, np.ndarray):
		if x.ndim == 3:
			x = np.expand_dims(x, axis=0)
			x = np.expand_dims(x, axis=0)
		elif x.ndim == 4:
			x = np.expand_dims(x, axis=0)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
		return to_5d_array(x=x)
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		if all(3 <= x_.ndim <= 4 for x_ in x):
			return to_5d_array(x=np.stack(x))
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
		
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		x = [_x.detach().cpu().numpy() for _x in x]
		return to_5d_array(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_5d_array(x=x)
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_4d_arraylist(x: Union[Tensors, Arrays]) -> list[torch.Tensor]:
	"""Convert to a 4D-array list."""
	if isinstance(x, np.ndarray):
		if x.ndim == 3:
			x = np.expand_dims(x, axis=0)
			x = [x]
		elif x.ndim == 4:
			x = [x]
		elif x.ndim == 5:
			x = list(x)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
		return to_4d_arraylist(x=x)
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		if all(x_.ndim == 3 for x_ in x):
			x = np.stack(x, axis=0)
			x = [x]
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
	
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		x = [_x.detach().cpu().numpy() for _x in x]
		return to_4d_arraylist(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_4d_arraylist(x=x)
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


def to_4d_tensorlist(x: Union[Tensors, Arrays]) -> list[np.ndarray]:
	"""Convert to a 4D-tensor list."""
	if isinstance(x, torch.Tensor):
		if x.dim() == 3:
			x = x.unsqueeze(dim=0)
			x = [x]
		elif x.dim() == 4:
			x = [x]
		elif x.dim() == 5:
			x = list(x)
		elif x.ndim < 3:
			raise ValueError(f"Wrong dimension: x.dim={x.ndim}.")
		return x
	
	if isinstance(x, np.ndarray):
		return to_4d_tensorlist(x=torch.from_numpy(x))
	
	if isinstance(x, tuple):
		x = list(x)
	
	if isinstance(x, list) and is_list_of(x, torch.Tensor):
		if all(x_.ndim == 3 for x_ in x):
			x = torch.stack(x, dim=0)
			x = [x]
		elif all(x_.ndim < 3 for x_ in x):
			raise ValueError(f"Wrong dimension: x.dim={x[0].ndim}.")
		return x
	
	if isinstance(x, list) and is_list_of(x, np.ndarray):
		x = [torch.from_numpy(_x) for _x in x]
		return to_4d_tensorlist(x=x)
	
	if isinstance(x, dict):
		x = [v for k, v in x.items()]
		return to_4d_tensorlist(x=x)
	
	raise ValueError(f"Wrong type: type(x)={type(x)}.")


# MARK: - Modify

def slice_list(in_list: list, lens: Union[int, list]) -> list[list]:
	"""Slice a list into several sub lists by a list of given length.
	
	Args:
		in_list (list):
			The list to be sliced.
		lens(int, list):
			The expected length of each out list.
	
	Returns:
		out_list (list):
			A list of sliced list.
	"""
	if isinstance(lens, int):
		assert len(in_list) % lens == 0
		lens = [lens] * int(len(in_list) / lens)
	if not isinstance(lens, list):
		raise TypeError("`indices` must be an integer or a list of integers.")
	elif sum(lens) != len(in_list):
		raise ValueError(f'Sum of lens and list length does not match: {sum(lens)} != {len(in_list)}.')
	
	out_list = []
	idx      = 0
	for i in range(len(lens)):
		out_list.append(in_list[idx:idx + lens[i]])
		idx += lens[i]
	return out_list


def concat_list(in_list: list) -> list:
	"""Concatenate a list of list into a single list."""
	return list(itertools.chain(*in_list))


@dispatch(list)
def unique(in_list: list) -> list:
	"""Return a list with only unique elements."""
	return list(set(in_list))


@dispatch(tuple)
def unique(in_tuple: tuple) -> tuple:
	"""Return a tuple with only unique elements."""
	return tuple(set(in_tuple))
