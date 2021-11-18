#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional
from typing import Union

import torch.nn as nn
from munch import Munch
from torch import optim
from torch.optim import Optimizer
from torchkit.core.utils import is_list_of

from torchkit.core.factory import Registry

logger = logging.getLogger()


# MARK: - OptimFactory

class OptimFactory(Registry):
	"""The factory class for creating optimizers."""
	
	# MARK: Build
	
	def build(
		self, net: nn.Module, name: str, *args, **kwargs
	) -> Optional[Optimizer]:
		"""Factory command to create an optimizer. This method gets the
		appropriate optimizer class from the registry and creates an instance
		of it, while passing in the parameters given in `kwargs`.
		
		Args:
			net (nn.Module):
				The neural network module.
			name (str):
				The optimizer's name.
		
		Returns:
			instance (Optimizer, optional):
				An instance of the optimizer that is created.
		"""
		if name not in self.registry:
			logger.warning(f"{name} does not exist in the registry.")
			return None
		
		return self.registry[name](params=net.parameters(), *args, **kwargs)
	
	def build_from_dict(
		self, net: nn.Module, cfg: Optional[Union[dict, Munch]], **kwargs
	) -> Optional[Optimizer]:
		"""Factory command to create an optimizer. This method gets the
		appropriate optimizer class from the registry and creates an instance
		of it, while passing in the parameters given in `cfg`.

		Args:
			net (nn.Module):
				The neural network module.
			cfg (dict, Munch, optional):
				The optimizer' config.

		Returns:
			instance (Optimizer, optional):
				An instance of the optimizer that is created.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			logger.warning("`cfg` must be a dict.")
			return None
		if "name" not in cfg:
			logger.warning("The `cfg` dict must contain the key `name`.")
			return None
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(net=net, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		net : nn.Module,
		cfgs: Optional[list[Union[dict, Munch]]],
		**kwargs
	) -> Optional[list[Optimizer]]:
		"""Factory command to create optimizers. This method gets the
		appropriate optimizers classes from the registry and creates
		instances of them, while passing in the parameters given in `cfgs`.

		Args:
			net (nn.Module):
				The list of neural network modules.
			cfgs (list[dict, Munch], optional):
				The list of optimizers' configs.

		Returns:
			instance (list[Optimizer], optional):
				Instances of the optimizers that are created.
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
			instances.append(self.build(net=net, name=name, **cfg))
		
		return instances if len(instances) > 0 else None
	
	def build_from_list(
		self,
		nets: list[nn.Module],
		cfgs: Optional[list[Union[dict, Munch]]],
		*args, **kwargs
	) -> Optional[list[Optimizer]]:
		"""Factory command to create optimizers. This method gets the
		appropriate optimizers classes from the registry and creates
		instances of them, while passing in the parameters given in `cfgs`.

		Args:
			nets (list[nn.Module]):
				The list of neural network modules.
			cfgs (list[dict, Munch]):
				The list of optimizers' configs.

		Returns:
			instance (list[Optimizer], optional):
				Instances of the optimizers that are created.
		"""
		if cfgs is None:
			return None
		if (not is_list_of(cfgs, expected_type=dict) or
			not is_list_of(cfgs, expected_type=Munch)):
			raise TypeError("`cfgs` must be a list of dict.")
		if not is_list_of(nets, expected_type=dict):
			raise TypeError("`nets` must be a list of nn.Module.")
		assert len(nets) == len(cfgs), \
			f"The length of `nets` and `cfgs` must be the same."
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for net, cfg in zip(nets, cfgs_):
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(net=net, name=name, **cfg))
		
		return instances if len(instances) > 0 else None


# MARK: - Builder

OPTIMIZERS = OptimFactory(name="optimizers")

OPTIMIZERS.register(name="adadelta",    module=optim.Adadelta)
OPTIMIZERS.register(name="adagrad",     module=optim.Adagrad)
OPTIMIZERS.register(name="adam",        module=optim.Adam)
OPTIMIZERS.register(name="adamax",      module=optim.Adamax)
OPTIMIZERS.register(name="adam_w",      module=optim.AdamW)
OPTIMIZERS.register(name="asgd",        module=optim.ASGD)
OPTIMIZERS.register(name="lbfgs",       module=optim.LBFGS)
OPTIMIZERS.register(name="rms_prop",    module=optim.RMSprop)
OPTIMIZERS.register(name="r_prop",      module=optim.Rprop)
OPTIMIZERS.register(name="sgd",         module=optim.SGD)
OPTIMIZERS.register(name="sparse_adam", module=optim.SparseAdam)
