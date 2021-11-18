#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculates the model accuracy.
"""

from __future__ import annotations

import logging
from numbers import Number
from typing import Union

import numpy as np
import torch
from torch import nn

from torchkit.core.utils import ScalarOrTupleAnyT
from .builder import METRICS

logger = logging.getLogger()


# MARK: - Accuracy

def accuracy_numpy(
	y_hat     : np.ndarray,
	y         : np.ndarray,
	top_k     : ScalarOrTupleAnyT[int]    = 1,
	thresholds: ScalarOrTupleAnyT[Number] = 0.0,
) -> np.ndarray:
	"""Calculate accuracy according to the prediction and target.
	
	Args:
		y_hat (np.ndarray):
			The prediction.
		y (np.ndarray):
			The ground truth label of the prediction.
		top_k (ScalarOrTupleAnyT[int]):
			If the predictions in `top_k` matches the target, the predictions
			will be regarded as correct ones.
		thresholds (ScalarOrTupleAnyT[Number]):
			Predictions with scores under the thresholds are considered
			negative.
	
	Returns:
		accuracy (float, list[float], list[list[float]]):
			- float            : If both `top_k` and `thresholds` is a single
			 					 value.
			- list[float]      : If one of `top_k` or  `thresholds` is a tuple.
			- list[list[float]]: If both `top_k` and `thresholds` is a tuple.
								 And the first dim is `top_k`, the second
								 dim is `thresholds`.
	"""
	if isinstance(thresholds, Number):
		thresholds    = (thresholds, )
		result_single = True
	elif isinstance(thresholds, tuple):
		result_single = False
	else:
		raise TypeError(
			f"`thresholds` should be a number or tuple, but got "
			f"{type(thresholds)}."
		)

	results    = []
	max_k      = max(top_k)
	num        = y_hat.shape[0]
	pred_label = y_hat.argsort(axis=1)[:, -max_k:][:, ::-1]
	pred_score = np.sort(y_hat, axis=1)[:, -max_k:][:, ::-1]

	for k in top_k:
		correct_k         = pred_label[:, :k] == y.reshape(-1, 1)
		results_threshold = []
		for thr in thresholds:
			# Only prediction values larger than `thr` are counted as correct
			_correct_k = correct_k & (pred_score[:, :k] > thr)
			_correct_k = np.logical_or.reduce(_correct_k, axis=1)
			results_threshold.append(_correct_k.sum() * 100.0 / num)
		if result_single:
			results.append(results_threshold[0])
		else:
			results.append(results_threshold)

	return results


def accuracy_torch(
	y_hat     : torch.Tensor,
	y         : torch.Tensor,
	top_k     : ScalarOrTupleAnyT[int]    = 1,
	thresholds: ScalarOrTupleAnyT[Number] = 0.0,
) -> torch.Tensor:
	"""Calculate accuracy according to the prediction and target.
	
	Args:
		y_hat (torch.Tensor):
			The prediction.
		y (torch.Tensor):
			The ground truth label of the prediction.
		top_k (ScalarOrTupleAnyT[int]):
			If the predictions in `top_k` matches the target, the predictions
			will be regarded as correct ones.
		thresholds (ScalarOrTupleAnyT[Number]):
			Predictions with scores under the thresholds are considered
			negative.
	
	Returns:
		accuracy (float, list[float], list[list[float]]):
			- float            : If both `top_k` and `thresholds` is a single
								 value.
			- list[float]      : If one of `top_k` or  `thresholds` is a tuple.
			- list[list[float]]: If both `top_k` and `thresholds` is a tuple.
								 And the first dim is `top_k`, the second dim
								 is `thresholds`.
	"""
	if isinstance(thresholds, Number):
		thresholds    = (thresholds, )
		result_single = True
	elif isinstance(thresholds, tuple):
		result_single = False
	else:
		raise TypeError(
			f"`thresholds` should be a number or tuple, but got "
			f"{type(thresholds)}."
		)

	max_k = max(top_k)
	pred_score, pred_label = y_hat.topk(max_k, dim=1)

	pred_label = pred_label.t()
	correct    = pred_label.eq(y.view(1, -1).expand_as(pred_label))
	results    = []
	num        = y_hat.size()[0]

	for k in top_k:
		results_threshold = []
		for thr in thresholds:
			# Only prediction values larger than thr are counted as correct
			_correct  = correct & (pred_score.t() > thr)
			correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
			results_threshold.append(correct_k.mul_(100.0 / num))
		if result_single:
			results.append(results_threshold[0])
		else:
			results.append(results_threshold)

	return results


def accuracy(
	y_hat     : Union[torch.Tensor, np.ndarray],
	y         : Union[torch.Tensor, np.ndarray],
	top_k     : ScalarOrTupleAnyT[int]    = 1,
	thresholds: ScalarOrTupleAnyT[Number] = 0.0,
) -> Union[torch.Tensor, np.ndarray]:
	"""Calculate accuracy according to the predictions and targets.
	
	Args:
		y_hat (torch.Tensor, np.ndarray):
			The prediction.
		y (torch.Tensor, np.ndarray):
			The ground truth label of the prediction.
		top_k (ScalarOrTupleAnyT[int]):
			If the predictions in `top_k` matches the target, the predictions
			will be regarded as correct ones.
		thresholds (ScalarOrTupleAnyT[Number]):
			Predictions with scores under the thresholds are considered
			negative.
	
	Returns:
		accuracy (float, list[float], list[list[float]]):
			- float            : If both `top_k` and `thresholds` is a single
								 value.
			- list[float]      : If one of `top_k` or  `thresholds` is a tuple.
			- list[list[float]]: If both `top_k` and `thresholds` is a tuple.
								 And the first dim is `top_k`, the second dim
								 is `thresholds`.
	"""
	assert isinstance(top_k, (int, tuple))
	if isinstance(top_k, int):
		top_k         = (top_k, )
		return_single = True
	else:
		return_single = False

	if isinstance(y_hat, torch.Tensor) and isinstance(y, torch.Tensor):
		res = accuracy_torch(y_hat, y, top_k, thresholds)
	elif isinstance(y_hat, np.ndarray) and isinstance(y, np.ndarray):
		res = accuracy_numpy(y_hat, y, top_k, thresholds)
	else:
		raise TypeError(
			f"`y_hat` and `y` should both be `torch.Tensor` or "
			f"`np.ndarray`, but got {type(y_hat)} and {type(y)}."
		)

	return res[0] if return_single else res


@METRICS.register(name="accuracy", force=True)
class Accuracy(nn.Module):
	"""Module to calculate the accuracy.
	
	Attributes:
		top_k (ScalarOrTupleAnyT[int]):
			The criterion used to calculate the accuracy.
		name (str):
			Name of the metric.
	"""

	# MARK: Magic Functions

	def __init__(self, top_k: ScalarOrTupleAnyT[int] = (1, )):
		super().__init__()
		self.name  = "accuracy"
		self.top_k = top_k

	# MARK: Forward Pass

	def forward(
		self,
		y_hat: Union[torch.Tensor, np.ndarray],
		y    : Union[torch.Tensor, np.ndarray],
	) -> Union[torch.Tensor, np.ndarray]:
		acc = accuracy(y_hat, y, self.top_k)
		if len(self.top_k) == 1:
			return acc[0]
		return acc
