#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Add-in to the `pytorch_lightning.Trainer` class.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl

logger = logging.getLogger()


# MARK: - Trainer

class Trainer(pl.Trainer):
	"""Add several methods and properties to the trainer class."""
	
	# MARK: Properties
	
	@pl.Trainer.current_epoch.setter
	def current_epoch(self, current_epoch: int):
		self.fit_loop.current_epoch = current_epoch
	
	@pl.Trainer.global_step.setter
	def global_step(self, global_step: int):
		self.fit_loop.global_step = global_step
