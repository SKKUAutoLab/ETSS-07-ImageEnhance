#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base model for all models defined in `torchkit.models`
"""

from __future__ import annotations

import logging
import os
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
from munch import Munch
from torch import nn

from torchkit.core.data import ClassLabels
from torchkit.core.fileio import create_dirs
from torchkit.core.fileio import filedir
from torchkit.core.fileio import is_url_or_file
from torchkit.core.loss import LOSSES
from torchkit.core.metric import METRICS
from torchkit.core.optim import OPTIMIZERS
from torchkit.core.scheduler import SCHEDULERS
from torchkit.core.utils import Arrays
from torchkit.core.utils import Dim3
from torchkit.core.utils import ForwardOutput
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import Indexes
from torchkit.core.utils import Metrics
from torchkit.core.utils import Tensors
from torchkit.utils import checkpoints_dir
from torchkit.utils import models_zoo_dir
from .debugger import Debugger
from .model_io import load_pretrained
from .utils import get_next_version

logger = logging.getLogger()


# MARK: - Phase

class Phase(Enum):
	"""The 3 basic phases of the model: `training`, `testing`, `inference`.
	"""
	
	# Produce predictions, calculate losses and metrics, update weights at the
	# end of each epoch/step.
	TRAINING  = "training"
	# Produce predictions, calculate losses and metrics, DO NOT update weights
	# at the end of each epoch/step.
	TESTING   = "testing"
	# Produce predictions ONLY.
	INFERENCE = "inference"
	
	@staticmethod
	def values() -> list[str]:
		"""Return the list of all values."""
		return [e.value for e in Phase]
	
	@staticmethod
	def keys():
		"""Return the list of all enum keys."""
		return [e for e in Phase]
	

# MARK: - BaseModel

# noinspection PyAttributeOutsideInit,PyMethodMayBeStatic
class BaseModel(pl.LightningModule, metaclass=ABCMeta):
	"""Base model for all models defined in `torchkit.models`. The base model
	only provides access to the attributes. In the model, each head is
	responsible for generating the appropriate output with accommodating loss
	and metric (obviously, we can only calculate specific loss and metric
	with specific output type). So we define the loss functions and metrics
	in the head implementation instead of the model.
	
	Attributes:
		name (str):
			The model name. In case `None` is given, it will be
			`self.__class__.__name__`. Default: `None`.
		fullname (str, optional):
			The model fullname in the following format:
			{name}_{data_name}_{postfix}. Default: `None`.
		model_zoo (dict):
			A dictionary of all pretrained weights of the model.
		model_dir (str):
			The model's dir. Default: `None`.
		version_dir (str):
			The experiment version dir.
		weights_dir (str):
			The weights's weights dir.
		debug_dir (str):
			The debug output dir.
		shape (Dim3, optional):
			The image shape as [H, W, C].
		num_classes (int, optional):
			Number of classes for classification. Default: `None`.
		classlabels (ClassLabels, optional):
			The `ClassLabels` object that contains all labels in the dataset.
		out_indexes (Indexes):
			The list of output tensors taken from specific layers' indexes.
			If `>= 0`, return the ith layer's output.
			If `-1`, return the final layer's output. Default: `-1`.
		phase (Phase):
			The model's running phase.
		pretrained (bool, str, dict):
			Initialize weights from pretrained.
			- If `True`, use the original pretrained described by the author
			  (usually, ImageNet or COCO). By default, it is the first
			  element in the `model_urls` dictionary.
			- If `str` and is a file/path, then load weights from saved file.
			- In each inherited model, `pretrained` can be a dictionary's key
			  to get the corresponding local file or url of the weight.
		loss (nn.Module):
			The loss computation module.
		metrics (list[nn.Module]):
			The metrics computation modules.
		optims_cfgs (dict, list[dict], optional):
			A dictionary or a list dictionaries of optimizers' configs.
			Default: `None`.
		schedulers_cfgs (dict, list[dict], list[list[dict]], optional):
			A dictionary or a list dictionaries of schedulers' configs.
			Default: `None`.
		debug (dict, optional):
			The debug configs. Default: `None`.
		epoch_step (int):
			The current step in the epoch. It can be shared between train,
			validation, test, and predict. Mostly used for debugging purpose.
	"""
	
	model_zoo = {}
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		name       : Optional[str]          	 = None,
		fullname   : Optional[str]           	 = None,
		model_dir  : Optional[str]           	 = None,
		version    : Optional[Union[int, str]]   = None,
		shape      : Optional[Dim3]          	 = None,
		num_classes: Optional[int] 			     = None,
		classlabels: Optional[ClassLabels]   	 = None,
		out_indexes: Indexes 				     = -1,
		phase      : Phase                   	 = Phase.TRAINING,
		pretrained : Union[bool, str, dict]      = False,
		loss   	   : Optional[dict]	      		 = None,
		metrics	   : Optional[list[dict]]		 = None,
		optimizers : Optional[Union[dict, list]] = None,
		schedulers : Optional[Union[dict, list]] = None,
		debugger   : Optional[dict]              = None,
		*args, **kwargs
	):
		"""

		Args:
			name (str, optional):
				The model name. In case of `None`, it will be
				`self.__class__.__name__`.
			fullname (str, optional):
				The model fullname in the following format:
				{name}_{data_name}_{postfix}. In case of `None`, it will be
				`self.name`.
			version (int, str, optional):
				The Experiment version. If version is not specified the logger
				inspects the save directory for existing versions, then
				automatically assigns the next available version. If it is a
				string then it is used as the run-specific subdirectory name,
				otherwise `version_${version}` is used.
			optimizers (dict, list, optional):
			   A dictionary or a list dictionaries of optimizers' configs.
			   Default: `None`.
			debug (dict, list, optional):
			   A dictionary or a list dictionaries of schedulers' configs.
			   Default: `None`.
			debug (dict, optional):
				The debug's configs. Default: `None`.
		"""
		super().__init__(*args, **kwargs)
		self.name            = name
		self.fullname        = fullname
		self.shape           = shape
		self.num_classes	 = num_classes
		self.classlabels     = classlabels
		self.out_indexes     = out_indexes
		self.phase           = phase
		self.pretrained 	 = pretrained
		self.loss 			 = loss
		self.metrics 		 = metrics
		self.optims_cfgs     = optimizers
		self.optims			 = None
		self.schedulers_cfgs = schedulers
		self.schedulers      = None
		self.debugger 		 = None
		self.epoch_step		 = 0
		
		self.init_num_classes()
		self.init_debugger(debugger_cfg=Munch.fromDict(debugger))
		self.init_dirs(model_dir=model_dir, version=version)
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""Return the model name. For instance: `yolov5_coco_1920`:
			- The model `name` is `yolov5`.
			- The dataset is `coco`.
			- The postfix is `1920`.
		"""
		return self._name
	
	@name.setter
	def name(self, name: Optional[str] = None):
		"""Assign the model name.

		Args:
			name (str, optional):
				The model name. In case `None` is given, it will be
				`self.__class__.__name__`. Default: `None`.
		"""
		self._name = (name if (name is not None and name != "")
					  else self.__class__.__name__.lower())
	
	@property
	def fullname(self) -> str:
		"""Return the model fullname in the following format:
		{name}_{data_name}_{postfix}.
		For instance: `yolov5_coco_1920`:
			- The model `name` is `yolov5`.
			- The dataset is `coco`.
			- The postfix is `1920`.
		"""
		return self._fullname
	
	@fullname.setter
	def fullname(self, fullname: Optional[str] = None):
		"""Assign the model's fullname in the following format:
		{name}_{data_name}_{postfix}. In case of `None`, it will
		be `self.name`.s
			For instance: `yolov5_coco_1920`:
				- The model `name` is `yolov5`.
				- The dataset is `coco`.
				- The postfix is `1920`.
		
		Args:
			fullname (str, optional):
				The model fullname in the following format:
				{name}_{data_name}_{postfix}. Default: `None`.
		"""
		self._fullname = (fullname if (fullname is not None and fullname != "")
						  else self.name)
	
	@property
	def size(self) -> Optional[Dim3]:
		"""Return the model input dimension as [C, H, W]."""
		if self.shape is None:
			return None
		return self.shape[2], self.shape[0], self.shape[1]
	
	@property
	def dim(self) -> Optional[int]:
		"""Return number of dimensions for model's input."""
		if self.size is None:
			return None
		return len(self.size)
	
	@property
	def ndim(self) -> Optional[int]:
		"""Alias of `self.dim()`."""
		return self.dim
	
	@property
	def phase(self) -> Phase:
		"""Returns the model's running phase."""
		return self._phase
	
	@phase.setter
	def phase(self, phase: Phase = Phase.TRAINING):
		"""Assign the model's running phase."""
		self._phase = phase
		if self._phase is Phase.TRAINING:
			self.unfreeze()
		else:
			self.freeze()
	
	@property
	def pretrained(self) -> Optional[dict]:
		"""Returns the model's pretrained metadata."""
		return self._pretrained
	
	@pretrained.setter
	def pretrained(self, pretrained: Union[bool, str, dict] = False):
		"""Assign model's pretrained.
		
		Args:
			pretrained (bool, str, dict):
				Initialize weights from pretrained.
				- If `True`, use the original pretrained described by the
				  author (usually, ImageNet or COCO). By default, it is the
				  first element in the `model_urls` dictionary.
				- If `str` and is a file/path, then load weights from saved
				  file.
				- In each inherited model, `pretrained` can be a dictionary's
				  key to get the corresponding local file or url of the weight.
		"""
		if pretrained is True and len(self.model_zoo):
			self._pretrained = list(self.model_zoo.values())[0]
		elif pretrained in self.model_zoo:
			self._pretrained = self.model_zoo[pretrained]
		else:
			self._pretrained = None
		
		# Update num_classes if it is currently `None`.
		if (self._pretrained and self.num_classes is None and
			"num_classes" in self._pretrained):
			self.num_classes = self._pretrained["num_classes"]
			
	@property
	def debug_image_dir(self) -> str:
		"""Return the debug image dir path located at: <debug_dir>/<dir>."""
		debug_dir = os.path.join(
			self.debug_dir, f"{self.phase.value}_{(self.current_epoch + 1):03d}"
		)
		create_dirs(paths=[debug_dir])
		return debug_dir
	
	@property
	def debug_image_filepath(self) -> str:
		"""Return the debug image filepath located at: <debug_dir>/"""
		save_dir = self.debug_dir
		if self.debugger:
			save_dir = (self.debug_image_dir if self.debugger.save_to_subdir
						else self.debug_dir)
		
		return os.path.join(
			save_dir,
			f"{self.phase.value}_"
			f"{(self.current_epoch + 1):03d}_"
			f"{(self.epoch_step + 1):06}.jpg"
		)
	
	@property
	def with_loss(self) -> bool:
		"""Return whether if the `loss` has been defined."""
		return hasattr(self, "loss") and self.loss is not None
	
	@property
	def loss(self) -> Optional[nn.Module]:
		"""Return the loss computation module."""
		return self._loss
	
	@loss.setter
	def loss(self, loss: Optional[dict]):
		self._loss = LOSSES.build_from_dict(cfg=loss)
		if self._loss:
			self._loss.cuda()
	
	@property
	def with_metrics(self) -> bool:
		"""Return whether if the `metrics` has been defined."""
		return hasattr(self, "metrics") and self.metrics is not None
	
	@property
	def metrics(self) -> Optional[list[nn.Module]]:
		"""Return the list of metric computation modules."""
		return self._metrics
	
	@metrics.setter
	def metrics(self, metrics: Optional[list[dict]]):
		self._metrics = METRICS.build_from_dictlist(cfgs=metrics)
		# Move to CUDA device
		if self._metrics:
			for metric in self._metrics:
				metric.cuda()
	
	# MARK: Configure
	
	def init_num_classes(self):
		"""Initialize num_classes."""
		if (
			self.classlabels is not None and
			self.num_classes != self.classlabels.num_classes()
		):
			self.num_classes = self.classlabels.num_classes()
	
	def init_debugger(self, debugger_cfg: Optional[dict]):
		"""Initialize debugger."""
		if debugger_cfg and debugger_cfg.run_in_parallel:
			self.debugger 			= Debugger(**debugger_cfg)
			self.debugger.show_func = self.show_results
			"""
			self.debug_queue  = Queue(maxsize=self.debug.queue_size)
			self.thread_debug = threading.Thread(
				target=self.show_results_parallel
			)
			"""
	
	def init_dirs(
		self, model_dir: Optional[str], version: Union[int, str, None]
	):
		"""Initialize directories.
		
		Args:
			model_dir (str, optional):
				The model's dir. Default: `None`.
			version (int, str, optional):
				The Experiment version. If version is not specified the logger
				inspects the save directory for existing versions, then
				automatically assigns the next available version. If it is a
				string then it is used as the run-specific subdirectory name,
				otherwise `version_${version}` is used.
		"""
		if model_dir is None:
			self.model_dir = os.path.join(checkpoints_dir, self.fullname)
		else:
			self.model_dir = model_dir
		
		if version is None:
			version = get_next_version(root_dir=self.model_dir)
		if isinstance(version, int):
			version = f"version_{version}"
		self.version = version.lower()
		
		self.version_dir = os.path.join(self.model_dir,   self.version)
		self.weights_dir = os.path.join(self.version_dir, "weights")
		self.debug_dir   = os.path.join(self.version_dir, "debugs")
	
	def load_pretrained(self):
		"""Load pretrained weights. It only loads the intersection layers of
		matching keys and shapes between current model and pretrained.
		"""
		if self.pretrained:
			load_pretrained(
				module	  = self,
				model_dir = models_zoo_dir,
				strict	  = False,
				**self.pretrained
			)
		elif is_url_or_file(self.pretrained):
			"""load_pretrained(
				self,
				path 	  = self.pretrained,
				model_dir = models_zoo_dir,
				strict	  = False
			)"""
			raise NotImplementedError(
				"This function has not been implemented yet."
			)
		else:
			logger.warning(f"Cannot load from pretrained: {self.pretrained}!")
			
	def configure_optimizers(self):
		"""Choose what optimizers and learning-rate schedulers to use in your
		optimization. Normally you’d need one. But in the case of GANs or
		similar you might have multiple.

		Notice:
			- This function is called when training begin. Hence, the `optims`
			  should have already been defined at this point. So we simply
			  return the `self.optims` attributes.
			- You can override this function to manually define the optimizers.

		Returns:
			optims (Optimizer, list[Optimizer], optional):
				The list of optimizers.
			schedulers (_LRScheduler, list[_LRScheduler], optional):
				The list of schedulers.
		"""
		# NOTE: Optimizers
		if self.optims_cfgs is not None:
			self.optims = OPTIMIZERS.build_from_dictlist(net=self,
														 cfgs=self.optims_cfgs)
			logger.info(f"Optimizers: \n{self.optims}.")
		else:
			self.optims = None
			logger.warning(f"No optimizers have been defined! Consider "
						   f"subclassing this function to manually define the "
						   f"optimizers.")
			
		# NOTE: Schedulers
		if self.schedulers_cfgs is not None and self.optims is not None:
			self.schedulers = SCHEDULERS.build_from_list(
				optimizers=self.optims, cfgs=self.schedulers_cfgs
			)
			logger.info(f"Schedulers: \n{self.schedulers}.")
			return self.optims, self.schedulers
		else:
			self.schedulers = None
			logger.warning(f"No schedulers have been defined! Consider "
						   f"subclassing this function to manually define the "
						   f"schedulers.")
			return self.optims
	
	# MARK: Forward Pass
	
	def forward(
		self, x: Tensors, y: Optional[Tensors] = None, *args, **kwargs
	) -> ForwardOutput:
		"""Forward pass. When `phase=INFERENCE`, only `x` is given. Otherwise,
		both `x` and `y` are given, hence, we compute the loss and metrics
		also.

		Args:
			x (Tensors):
				`x` contains the images tensor of shape [B, C, H, W].
			y (Tensors, optional):
				`y` contains the ground truth for each input.

		Returns:
			y_hat (Tensors):
				The final predictions.
			metrics (Metric, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		if (self.phase is Phase.INFERENCE) or (y is None):
			return self.forward_infer(x=x, *args, **kwargs)
		else:
			return self.forward_train(x=x, y=y, *args, **kwargs)
	
	@abstractmethod
	def forward_train(
		self, x: Tensors, y: Tensors, *args, **kwargs
	) -> ForwardXYOutput:
		"""Forward pass during training with both `x` and `y` are given.

		Args:
			x (Tensors):
				`x` contains the images tensor of shape [B, C, H, W].
			y (Tensors):
				`y` contains the ground truth for each input.

		Returns:
			y_hat (Tensors):
				The final predictions tensor.
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		pass
	
	@abstractmethod
	def forward_infer(self, x: Tensors, *args, **kwargs) -> Tensors:
		"""Forward pass during inference with only `x` is given.

        Args:
            x (Tensors):
                `x` contains the images tensor of shape [B, C, H, W].

        Returns:
            y_hat (Tensors, optional):
                The final predictions tensor.
        """
		pass
	
	def forward_features(
		self, x: torch.Tensor, out_indexes: Optional[Indexes] = None
	) -> Tensors:
		"""Forward pass for features extraction.

		Args:
			x (torch.Tensor):
				The input image.
			out_indexes (Indexes, optional):
				The list of layers' indexes to extract features. This is called
				in `forward_features()` and is useful when the model
				is used as a component in another model.
				- If is a `tuple` or `list`, return an array of features.
				- If is a `int`, return only the feature from that layer's
				  index.
				- If is `-1`, return the last layer's output.
				Default: `None`.
		"""
		pass
	
	# MARK: Training
	
	def on_fit_start(self):
		"""Called at the very beginning of fit."""
		filedir.create_dirs(paths=[self.model_dir, self.version_dir,
								   self.weights_dir, self.debug_dir])
		
		if self.debugger:
			self.debugger.run_routine_start()
			# self.thread_debug.start()
	
	def on_fit_end(self):
		"""Called at the very end of fit."""
		if self.debugger:
			self.debugger.run_routine_end()
			# self.debug_queue.put([None, None, None, None])
	
	def on_train_epoch_start(self):
		"""Called in the training loop at the very beginning of the epoch."""
		self.epoch_step = 0
	
	def training_step(
		self, batch: Any, batch_idx: int, *args, **kwargs
	) -> Optional[Metrics]:
		"""Training step.

		Args:
			batch (Any):
				The batch of inputs. It can be a tuple of
				(`x`, `y`, extra_info).
			batch_idx (int):
				The batch index.

		Returns:
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		# NOTE: Forward pass
		x, y, rest     = batch[0], batch[1], batch[2:]
		y_hat, metrics = self.forward(x=x, y=y, *args, **kwargs)
		
		# NOTE: Log loss and metrics
		self.parse_metrics(metrics=metrics, prefix="train")
		
		self.epoch_step += 1
		return metrics
	
	def on_validation_epoch_start(self):
		"""Called in the validation loop at the very beginning of the epoch."""
		self.epoch_step = 0
		
	def validation_step(
		self, batch: Any, batch_idx: int, *args, **kwargs
	) -> Optional[Metrics]:
		"""Validation step.

		Args:
			batch (Any):
				The batch of inputs. It can be a tuple of
				(`x`, `y`, extra_info).
			batch_idx (int):
				The batch index.

		Returns:
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		# NOTE: Forward pass
		x, y, rest     = batch[0], batch[1], batch[2:]
		y_hat, metrics = self.forward(x=x, y=y, *args, **kwargs)
		
		# NOTE: Log loss and metrics
		self.parse_metrics(metrics=metrics, prefix="val")
		
		# NOTE: Debugging
		epoch = self.current_epoch + 1
		if (self.debugger and epoch % self.debugger.every_n_epochs == 0 and
			self.epoch_step < self.debugger.save_max_n):
			if self.trainer.is_global_zero:
				self.debugger.run(
					deepcopy(x), deepcopy(y), deepcopy(y_hat),
					self.debug_image_filepath
				)
				"""if self.thread_debug:
					self.debug_queue.put([
						deepcopy(x), deepcopy(y), deepcopy(y_hat),
						self.debug_image_filepath
					])
				else:
					self.show_results(x, y, y_hat, **self.debug)"""
		
		self.epoch_step += 1
		return metrics
	
	def on_test_start(self) -> None:
		"""Called at the very beginning of testing."""
		self.create_dirs()
		
	def on_test_epoch_start(self) -> None:
		"""Called in the test loop at the very beginning of the epoch."""
		self.epoch_step = 0
	
	def test_step(
		self, batch: Any, batch_idx: int, *args, **kwargs
	) -> Optional[Metrics]:
		"""Test step.

		Args:
			batch (Any):
				The batch of inputs. It can be a tuple of
				(`x`, `y`, extra_info).
			batch_idx (int):
				The batch index.

		Returns:
			metrics (Metrics, optional):
				- A dictionary with the first key must be the `loss`.
				- `None`, training will skip to the next batch.
		"""
		# NOTE: Forward pass
		x, y, rest     = batch[0], batch[1], batch[2:]
		y_hat, metrics = self.forward(x=x, y=y, *args, **kwargs)
		
		# NOTE: Log loss and metrics
		self.parse_metrics(metrics=metrics, prefix="test")
		
		self.epoch_step += 1
		return metrics
	
	# MARK: Prediction
	
	def predict_step(
		self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
	) -> Union[Tensors, Arrays]:
		"""Predict step.

		Args:
			batch (Any):
				The batch of inputs. It can be a tuple of
				(`x`, `y`, extra_info).
			batch_idx (int):
				The batch index.
			dataloader_idx (int, optional):
				The dataloader index. Default: `None`.

		Returns:
			y_hat (Tensors, Arrays):
				The final predictions.
		"""
		x     = batch
		y_hat = self.forward(x=x)
		y_hat = self.prepare_results(y_hat=y_hat)
		return y_hat
	
	# MARK: Export
	
	def export_to_onnx(
		self,
		input_dims   : Optional[Dim3] = None,
		filepath     : Optional[str]  = None,
		export_params: bool           = True
	):
		"""Export the model to `onnx` format.

		Args:
			input_dims (Dim3, optional):
				The input dimensions. Default: `None`.
			filepath (str, optional):
				The path to save the model. If `None` or empty, then save to
				`zoo_dir`. Default: `None`.
			export_params (bool):
				Should export parameters also? Default: `True`.
		"""
		# NOTE: Check filepath
		if filepath in [None, ""]:
			filepath = os.path.join(self.version_dir, f"{self.fullname}.onnx")
		if ".onnx" not in filepath:
			filepath += ".onnx"
		
		if input_dims is not None:
			input_sample = torch.randn(input_dims)
		elif self.dims is not None:
			input_sample = torch.randn(self.dims)
		else:
			raise ValueError(f"No input dims are defined.")
		
		self.to_onnx(filepath=filepath, input_sample=input_sample,
					 export_params=export_params)
	
	def export_to_torchscript(
		self,
		input_dims: Optional[Dim3] = None,
		filepath  : Optional[str]  = None,
		method    : str            = "script"
	):
		"""Export the model to `TorchScript` format.

		Args:
			input_dims (Dim3, optional):
				The input dimensions.
			filepath (str, optional):
				The path to save the model. If `None` or empty, then save
				to `zoo_dir`. Default: `None`.
			method (str):
				Whether to use TorchScript's `script` or `trace` method.
				Default: `script`
		"""
		# NOTE: Check filepath
		if filepath in [None, ""]:
			filepath = os.path.join(self.version_dir, f"{self.fullname}.pt")
		if ".pt" not in filepath:
			filepath += ".pt"
		
		if input_dims is not None:
			input_sample = torch.randn(input_dims)
		elif self.dims is not None:
			input_sample = torch.randn(self.dims)
		else:
			raise ValueError(f"No input dims are defined.")
		
		script = self.to_torchscript(method=method, example_inputs=input_sample)
		torch.jit.save(script, filepath)

	# MARK: Visualize
	
	"""
	def show_results_parallel(self):
		while True:
			(x, y, y_hat, filepath) = self.debug_queue.get()
			if x is None:
				break
			
			self.show_results(
				x=x, y=y, y_hat=y_hat, filepath=filepath, **self.debug
			)
		
		# Stop debugger thread
		self.thread_debug.join()
	"""
	
	@abstractmethod
	def show_results(
		self,
		x    		 : Union[Tensors, Arrays],
		y    		 : Optional[Union[Tensors, Arrays]] = None,
		y_hat		 : Optional[Union[Tensors, Arrays]] = None,
		filepath	 : Optional[str] 				    = None,
		image_quality: int             				    = 95,
		show         : bool             				= False,
		show_max_n   : int             					= 8,
		wait_time    : float            				= 0.01,
		*args, **kwargs
	):
		"""Draw `result` over input image.

		Args:
			x (Tensors, Arrays):
				The input images.
			y (Tensors, Arrays, optional):
				The ground truth.
			y_hat (Tensors, Arrays, optional):
				The predictions.
			filepath (str, optional):
				The file path to save the debug result.
			image_quality (int):
				The image quality to be saved. Default: `95`.
			show (bool):
				If `True` shows the results on the screen. Default: `False`.
			show_max_n (int):
				Maximum debugging items to be shown. Default: `8`.
			wait_time (float):
				Pause some times before showing the next image.
		"""
		pass
	
	def prepare_results(
		self,
		y_hat: Union[Tensors, Arrays],
		x    : Optional[Union[Tensors, Arrays]] = None,
		y    : Optional[Union[Tensors, Arrays]] = None,
		*args, **kwargs
	) -> Arrays:
		"""Prepare results for visualization.

		Args:
			y_hat (Tensors, Arrays):
				The predictions.
			x (Tensors, Arrays, optional):
				The input images. Default: `None`.
			y (Tensors, Arrays, optional):
				The ground truth. Default: `None`.

		Returns:
			results (Arrays):
				The results for visualization.
		"""
		return y_hat
	
	# MARK: Utils

	def parse_metrics(
		self, metrics: Optional[Metrics] = None, prefix: str = ""
	):
		"""Parse metrics.

		Args:
			metrics (Metrics, optional):
				A dictionary with the first key must be the `loss`.
				Default: `None`.
			prefix (str):
				The prefix for the metrics. Default: ``.
		"""
		if isinstance(metrics, dict):
			for i, (key, value) in enumerate(metrics.items()):
				if i == 0:
					assert "loss" in key, f"The first key must be `loss`."
				if prefix != "":
					name = f"{prefix}_{key}"
				else:
					name = f"{key}"
					
				# For top-k metrics, we only get the first one.
				if isinstance(value, (list, tuple)):
					value = value[0]
				self.log(f"{name}", value, on_step=True, on_epoch=True,
						 prog_bar=True, rank_zero_only=True)
