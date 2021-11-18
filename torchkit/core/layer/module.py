# ==================================================================== #
# File name: module.py
# Author: Long H. Pham
# Date created: 09/02/2021
# The `torchkit.core.nn.module` defines base modules for all modules
# in `torchkit`.
# ==================================================================== #
import copy
import logging
import re
import warnings
from abc import ABCMeta
from collections import defaultdict
from typing import Optional

import torch.nn as nn

logger = logging.getLogger()


# MARK: - Module

# noinspection PyTypeChecker
class Module(nn.Module, metaclass=ABCMeta):
	"""Base module for all modules. ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional functionality
	of parameter initialization. Compared with ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.
	- `init_cfg`:
			The config to control the initialization.
	- `init_weights`:
			The function of parameter initialization and recording initialization information.
	- `_params_init_info`:
			Used to track the parameter initialization information. This attribute only exists during executing the
			`init_weights`.
	
	Attributes:
		init_cfg (dict, optional):
			Initialization config dict. `init_cfg` can be defined in different levels, but `init_cfg` in low levels has
			a higher priority. Define default value of `init_cfg` instead of hard code in `init_weights()` function.
			Default: `None`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		name    : Optional[str]  = None,
		init_cfg: Optional[dict] = None,
		*args, **kwargs
	):
		"""Initialize BaseModule, inherited from `nn.Module`.
		
		Args:
			name (str, optional):
				Name of the module. Default: `None`.
			init_cfg (dict, optional):
				Initialization config dict. `init_cfg` can be defined in different levels, but `init_cfg` in low levels
				has a higher priority. Define default value of `init_cfg` instead of hard code in `init_weights()`
				function. Default: `None`.
		"""
		super().__init__()
		self._name    = name
		self._is_init = False
		self.init_cfg = copy.deepcopy(init_cfg)
	
	def __repr__(self):
		s = super().__repr__()
		if self.init_cfg:
			s += f"`init_cfg`={self.init_cfg}"
		return s
	
	# MARK: Properties
	
	@property
	def __name__(self) -> str:
		return self.name
	
	@property
	def name(self) -> str:
		"""Return the name of the module.
		
		Returns:
			name (str):
				Name of the module.
		"""
		if self._name is None:
			name = self.__class__.__name__
			s1   = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
			return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
		else:
			return self._name.lower()
	
	@property
	def is_init(self) -> bool:
		"""Return if the current module has been initialized yet.
		
		Returns:
			_is_init (bool):
				Has the current module been initialized yet.
		"""
		return self._is_init
	
	# MARK: Configure

	def freeze(self):
		""" Freeze all params for inference. An alias of `self.eval()`.
		"""
		for param in self.parameters():
			param.requires_grad = False
			
		return self.eval()
	
	def unfreeze(self):
		"""Unfreeze all parameters for training. An alias of `self.train()`.
		"""
		for param in self.parameters():
			param.requires_grad = True
			
		return self.train()
	

# MARK: - Sequential

class Sequential(Module, nn.Sequential):
	"""Sequential module.
	
	Attributes:
		init_cfg (dict, optional):
			Initialization config dict.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		name    : Optional[str]  = None,
		init_cfg: Optional[dict] = None,
		*args, **kwargs
	):
		"""
		
		Args:
			name (str, optional):
				Name of the module. Default: `None`.
			init_cfg (dict, optional):
				Initialization config dict. `init_cfg` can be defined in different levels, but `init_cfg` in low levels
				has a higher priority. Define default value of `init_cfg` instead of hard code in `init_weights()`
				function. Default: `None`.
		"""
		Module.__init__(self, name=name, init_cfg=init_cfg, *args, **kwargs)
		nn.Sequential.__init__(self, *args, **kwargs)


# MARK: - ModuleList

class ModuleList(Module, nn.ModuleList):
	"""ModuleList.
	
	Attributes:
		modules (iterable, optional):
			An iterable of modules to add.
		init_cfg (dict, optional):
			Initialization config dict.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		modules                  = None,
		name    : Optional[str]  = None,
		init_cfg: Optional[dict] = None,
		*args, **kwargs
	):
		"""
		
		Args:
			name (str, optional):
				Name of the module. Default: `None`.
			init_cfg (dict, optional):
				Initialization config dict. `init_cfg` can be defined in different levels, but `init_cfg` in low levels
				has a higher priority. Define default value of `init_cfg` instead of hard code in `init_weights()`
				function. Default: `None`.
		"""
		Module.__init__(
			self,
			name     = name,
			init_cfg = init_cfg,
			*args, **kwargs
		)
		nn.ModuleList.__init__(self, modules)
