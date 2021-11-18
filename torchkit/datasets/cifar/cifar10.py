#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CIFAR10 dataset and datamodule
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

from torchkit.core.data import ClassLabels
from torchkit.core.dataset import DataModule
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchkit.datasets.builder import DATAMODULES
from torchkit.utils import datasets_dir

__all__ = ["Cifar10DataModule"]


# MARK: - Data Config

data = {
	"name": "cifar10",
	# The dataset's name.
	"shape": [32, 32, 3],
	# The image shape as [H, W, C]. This is compatible with OpenCV format.
	# This is also used to reshape the input data.
	"num_classes": 10,
	# The number of classes in the dataset.
	"batch_size": 32,
	# Number of samples in one forward & backward pass.
	"shuffle": True,
	# Set to `True` to have the data reshuffled at every training epoch.
	# Default: `True`.
}


# MARK: - Cifar10DataModule

@DATAMODULES.register(name="cifar10")
class Cifar10DataModule(DataModule):
	"""Cifar10 DataModule.
	
	Examples:
		>> Cifar10DataModule(name="cifar10", shape=(32, 32, 3), batch_size=32,
		shuffle=True)
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dataset_dir: str = os.path.join(datasets_dir, "cifar"),
		name       : str = "cifar10",
		*args, **kwargs
	):
		super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)

		self.transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize(size=self.shape[0:2]),
			torchvision.transforms.ToTensor()
		])
	
	# MARK: Prepare Data
	
	def prepare_data(self, *args, **kwargs):
		"""Use this method to do things that might write to disk or that need
		to be done only from a single GPU in distributed settings.
			- Download.
			- Tokenize.
		"""
		CIFAR10(root=self.dataset_dir, train=True , download=True)
		CIFAR10(root=self.dataset_dir, train=False, download=True)
		if self.classlabels is None:
			self.load_classlabels()
	
	def setup(self, phase: Optional[Phase] = None):
		"""There are also data operations you might want to perform on every
		GPU. Use setup to do things like:
			- Count number of classes.
			- Build labels vocabulary.
			- Perform train/val/test splits.
			- Apply transforms (defined explicitly in your datamodule or
			  assigned in init).

		Args:
			phase (Phase, optional):
				The phase to use: [None, Phase.TRAINING, Phase.TESTING].
				Set to "None" to setup all train, val, and test data.
				Default: `None`.
		"""
		# NOTE: Assign train/val datasets for use in dataloaders
		if phase in [None, Phase.TRAINING]:
			full_dataset = CIFAR10(self.dataset_dir, train=True, download=True,
								   transform=self.transform)
			self.train, self.val = random_split(full_dataset, [45000, 5000])

		# NOTE: Assign test datasets for use in dataloader(s)
		if phase in [None, Phase.TESTING]:
			self.test = CIFAR10(self.dataset_dir, train=False, download=True,
								transform=self.transform)
		
		if self.classlabels is None:
			self.load_classlabels()
		
	def load_classlabels(self):
		current_dir = os.path.dirname(os.path.abspath(__file__))
		classlabel_path = os.path.join(
			current_dir, "cifar10_classlabels.json"
		)
		self.classlabels = ClassLabels.create_from_file(
			label_path=classlabel_path
		)
		

# MARK: - Main

if __name__ == "__main__":
	# NOTE: Get DataModule
	cfg = data
	dm  = Cifar10DataModule(**cfg)
	dm.setup()
	# NOTE: Visualize labels
	print(f"Classlabels \n{dm.classlabels.list}")
	# NOTE: Visualize an iteration
	data_iter 		= iter(dm.train_dataloader)
	images, targets = next(data_iter)
	show_images(images=images)
	plt.show(block=True)
