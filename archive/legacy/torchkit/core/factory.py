#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base factory class for creating and registering classes.
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from pprint import pprint
from typing import Optional
from typing import Union

from munch import Munch

from .utils import FuncCls
from .utils import is_list_of

logger = logging.getLogger()


# MARK: - Registry

class Registry:
	"""The base registry class for registering classes.

	Attributes:
		name (str):
			The registry name.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, name: str):
		self._name     = name
		self._registry = {}
	
	def __len__(self):
		return len(self._registry)
	
	def __contains__(self, key: str):
		return self.get(key) is not None
	
	def __repr__(self):
		format_str = self.__class__.__name__ \
					 + f"(name={self._name}, items={self._registry})"
		return format_str
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""Return the registry's name."""
		return self._name
	
	@property
	def registry(self) -> dict:
		"""Return the registry's dictionary."""
		return self._registry
	
	def get(self, key: str) -> FuncCls:
		"""Get the registry record of the given `key`."""
		if key in self._registry:
			return self._registry[key]
	
	# MARK: Register
	
	def register(
		self,
		name  : Optional[str] = None,
		module: FuncCls		  = None,
		force : bool          = False
	) -> callable:
		"""Register a module.

		A record will be added to `self._registry`, whose key is the class name
		or the specified name, and value is the class itself. It can be used
		as a decorator or a normal function.

		Example:
			# >>> backbones = Factory("backbone")
			# >>>
			# >>> @backbones.register()
			# >>> class ResNet:
			# >>>     pass
			# >>>
			# >>> @backbones.register(name="mnet")
			# >>> class MobileNet:
			# >>>     pass
			# >>>
			# >>> class ResNet:
			# >>>     pass
			# >>> backbones.register(ResNet)

		Args:
			name (str, optional):
				The module name to be registered. If not specified, the class
				name will be used.
			module (type):
				Module class to be registered.
			force (bool):
				Whether to override an existing class with the same name.
		"""
		if not (name is None or isinstance(name, str)):
			raise TypeError(
				f"`name` must be either of `None` or an instance of `str`, "
				f"but got {type(name)}."
			)
		
		# NOTE: Use it as a normal method: x.register(module=SomeClass)
		if module is not None:
			self.register_module(module, name, force)
			return module
		
		# NOTE: Use it as a decorator: @x.register()
		def _register(cls):
			self.register_module(cls, name, force)
			return cls
		
		return _register
	
	def register_module(
		self,
		module_class: FuncCls,
		module_name : Optional[str] = None,
		force	    : bool 			= False
	):
		if not inspect.isclass(module_class):
			raise TypeError(
				f"module must be a class, but got {type(module_class)}."
			)
		if module_name is None:
			module_name = module_class.__name__.lower()
		if isinstance(module_name, str):
			module_name = [module_name]
		for name in module_name:
			if not force and name in self._registry:
				logger.info(f"{name} is already registered in {self.name}.")
				continue
			self._registry[name] = module_class
	
	# MARK: Print

	def print(self):
		"""Print the registry dictionary."""
		print(f"{self.name}:")
		pprint(self.registry, sort_dicts=False)
		print()


# MARK: - Factory

class Factory(Registry):
	"""The default factory class for creating objects.
	
	Registered object could be built from registry.
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
	"""
	
	# MARK: Build
	
	def build(self, name: str, *args, **kwargs) -> object:
		"""Factory command to create an instance of the class. This method gets
		the appropriate class from the registry and creates an instance of
		it, while passing in the parameters given in `kwargs`.
		
		Args:
			name (str):
				The name of the class to create.
			
		Returns:
			instance (object, optional):
				An instance of the class that is created.
		"""
		if name not in self.registry:
			logger.warning(f"{name} does not exist in the registry.")
			return None
		
		return self.registry[name](*args, **kwargs)
	
	def build_from_dict(
		self, cfg: Optional[Union[dict, Munch]], **kwargs
	) -> Optional[object]:
		"""Factory command to create an instance of a class. This method gets
		the appropriate class from the registry while passing in the
		parameters given in `cfg`.
		
		Args:
			cfg (dict, Munch):
				The class object' config.
		
		Returns:
			instance (object, optional):
				An instance of the class that is created.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			logger.warning("`cfg` must be a dict.")
			return None
		if "name" not in cfg:
			logger.warning("The `cfg` dict must contain the key `name`.")
			return None
		
		cfg_    = deepcopy(cfg)
		name    = cfg_.pop("name")
		cfg_   |= kwargs
		return self.build(name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		cfgs: Optional[list[Union[dict, Munch]]],
		**kwargs
	) -> Optional[list[object]]:
		"""Factory command to create instances of classes. This method gets the
		appropriate classes from the registry while passing in the parameters
		given in `cfgs`.

		Args:
			cfgs (list[dict, Munch], optional):
				The list of class objects' configs.

		Returns:
			instances (list[object], optional):
				Instances of the classes that are created.
		"""
		if cfgs is None:
			return None
		if (not is_list_of(cfgs, expected_type=dict) or
			not is_list_of(cfgs, expected_type=Munch)):
			logger.warning("`cfgs` must be a list of dict.")
			return None
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(name=name, **cfg))
		
		return instances if len(instances) > 0 else None
