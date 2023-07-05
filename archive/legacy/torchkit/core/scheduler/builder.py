#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional
from typing import Union

from munch import Munch
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from torchkit.core.factory import Registry
from torchkit.core.utils import is_list_of

logger = logging.getLogger()


# MARK: - SchedulerFactory

class SchedulerFactory(Registry):
	"""The factory class for creating schedulers."""
	
	# MARK: Build
	
	def build(
        self, optimizer: Optimizer, name: Optional[str], *args, **kwargs
    ) -> Optional[_LRScheduler]:
		"""Factory command to create an scheduler. This method gets the
		appropriate scheduler class from the registry and creates an instance
		of it, while passing in the parameters given in `kwargs`.
		
		Args:
			optimizer (Optimizer):
				The optimizer.
			name (str, optional):
				The scheduler's name.
		
		Returns:
			instance (_LRScheduler, optional):
				An instance of the scheduler that is created.
		"""
		if name is None:
			return None
		if name not in self.registry:
			logger.warning(f"{name} does not exist in the registry.")
			return None
		
		if name in ["GradualWarmupScheduler", "gradual_warmup_scheduler"]:
			after_scheduler = kwargs.pop("after_scheduler")
			if isinstance(after_scheduler, dict):
				name_ = after_scheduler.pop("name")
				if name_ in self.registry:
					after_scheduler = self.registry[name_](optimizer=optimizer,
                                                           **after_scheduler)
				else:
					after_scheduler = None
			return self.registry[name](optimizer=optimizer,
                                       after_scheduler=after_scheduler,
                                       *args, **kwargs)
		
		return self.registry[name](optimizer=optimizer, *args, **kwargs)
	
	def build_from_dict(
		self,
		optimizer: Optimizer,
		cfg      : Optional[Union[dict, Munch]],
		*args, **kwargs
	) -> Optional[_LRScheduler]:
		"""Factory command to create an scheduler. This method gets the
		appropriate scheduler class from the registry and creates an
		instance of it, while passing in the parameters given in `cfg`.

		Args:
			optimizer (Optimizer):
				The optimizer.
			cfg (dict, Munch, optional):
				The scheduler' config.

		Returns:
			instance (_LRScheduler, optional):
				An instance of the scheduler that is created.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			raise TypeError("`cfg` must be a dict.")
		if "name" not in cfg:
			raise KeyError("The `cfg` dict must contain the key `name`.")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(optimizer=optimizer, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		optimizer: Optimizer,
		cfgs     : Optional[list[Union[dict, Munch]]],
		*args, **kwargs
	) -> Optional[list[_LRScheduler]]:
		"""Factory command to create schedulers. This method gets the
		appropriate schedulers classes from the registry and creates
		instances of them, while passing in the parameters given in `cfgs`.

		Args:
			optimizer (Optimizer):
				The optimizer.
			cfgs (list[dict, Munch], optional):
				The list of optimizers' configs.

		Returns:
			instance (list[Optimizer], optional):
				Instances of the scheduler that are created.
		"""
		if cfgs is None:
			return None
		if (
            not is_list_of(cfgs, expected_type=dict) or
            not is_list_of(cfgs, expected_type=Munch)
        ):
			raise TypeError("`cfgs` must be a list of dict.")
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(optimizer=optimizer, name=name, **cfg))
		
		return instances if len(instances) > 0 else None
	
	def build_from_list(
		self,
		optimizers: list[Optimizer],
		cfgs      : Optional[list[list[Union[dict, Munch]]]],
		*args, **kwargs
	) -> Optional[list[_LRScheduler]]:
		"""Factory command to create schedulers. This method gets the
		appropriate schedulers classes from the registry and creates
		instances of them, while passing in the parameters given in `cfgs`.

		Args:
			optimizers (list[Optimizer]):
				The list of optimizers.
			cfgs (list[list[dict, Munch]], optional):
				The 2D-list of optimizers' configs.

		Returns:
			instance (list[Optimizer], optional):
				Instances of the scheduler that are created.
		"""
		if cfgs is None:
			return None
		if (
            not is_list_of(cfgs, expected_type=list) or
            not all(is_list_of(cfg, expected_type=dict) for cfg in cfgs)
        ):
			raise TypeError("`cfgs` must be a 2D-list of dict.")
		assert len(optimizers) == len(cfgs)
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for optimizer, cfgs in zip(optimizers, cfgs_):
			for cfg in cfgs:
				name  = cfg.pop("name")
				cfg  |= kwargs
				instances.append(self.build(optimizer=optimizer, name=name,
											**cfg))
		
		return instances if len(instances) > 0 else None


# MARK: - Builder

SCHEDULERS = SchedulerFactory(name="schedulers")

SCHEDULERS.register(name="cosine_annealing_lr",            module=CosineAnnealingLR)
SCHEDULERS.register(name="cosine_annealing_warm_restarts", module=CosineAnnealingWarmRestarts)
SCHEDULERS.register(name="cyclic_lr",                      module=CyclicLR)
SCHEDULERS.register(name="exponential_lr",                 module=ExponentialLR)
SCHEDULERS.register(name="lambda_lr",                      module=LambdaLR)
SCHEDULERS.register(name="multistep_lr",                   module=MultiStepLR)
SCHEDULERS.register(name="reduce_lr_on_plateau",           module=ReduceLROnPlateau)
SCHEDULERS.register(name="step_lr",                        module=StepLR)
