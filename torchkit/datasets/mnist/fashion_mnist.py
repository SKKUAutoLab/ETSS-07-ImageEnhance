#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FashionMNIST dataset and datamodule.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import random_split
from torchkit.core.data import ClassLabels
from torchkit.core.dataset import DataModule
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchvision.datasets import FashionMNIST

from torchkit.datasets.builder import DATAMODULES
from torchkit.utils import datasets_dir
from torchkit.utils import load_config

__all__ = ["FashionMNIST", "FashionMNISTDataModule"]


# MARK: - Data Config

data = {
	"name": "fashion_mnist",
	# The dataset's name.
	"shape": [32, 32, 1],
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


# MARK: - FashionMNISTDataModule

@DATAMODULES.register(name="fashion_mnist")
class FashionMNISTDataModule(DataModule):
	"""FashionMNIST DataModule.

	Attributes:
		Same attributes as `torchkit.core.dataset.datamodule`.

	Examples:
		>> FashionMNISTDataModule(name="fashion_mnist", shape=(32, 32, 1),
		batch_size=32, id_level="id", shuffle=True)
	"""
   
	# MARK: Magic Functions
	
	def __init__(
		self,
		dataset_dir: str = os.path.join(datasets_dir, "mnist"),
		name       : str = "fashion_mnist",
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
		FashionMNIST(root=self.dataset_dir, train=True,  download=True)
		FashionMNIST(root=self.dataset_dir, train=False, download=True)
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
			full_dataset = FashionMNIST(
				self.dataset_dir,
				train     = True,
				download  = True,
				transform = self.transform
			)
			self.train, self.val = random_split(full_dataset, [55000, 5000])

		# NOTE: Assign test datasets for use in dataloader(s)
		if phase in [None, Phase.TESTING]:
			self.test = FashionMNIST(
				self.dataset_dir,
				train     = False,
				download  = True,
				transform = self.transform
			)
		
		if self.classlabels is None:
			self.load_classlabels()
	
	def load_classlabels(self):
		current_dir = os.path.dirname(os.path.abspath(__file__))
		classlabel_path = os.path.join(
			current_dir, "fashion_mnist_classlabels.json"
		)
		self.classlabels = ClassLabels.create_from_file(
			label_path=classlabel_path
		)


# MARK: - Main

if __name__ == "__main__":
	# NOTE: Get DataModule
	cfg = data
	dm  = FashionMNISTDataModule(**cfg)
	dm.setup()
	# NOTE: Visualize labels
	print(f"Classlabels \n{dm.classlabels.list}")
	# NOTE: Visualize an iteration
	data_iter       = iter(dm.train_dataloader)
	images, targets = next(data_iter)
	show_images(images=images)
	plt.show(block=True)
