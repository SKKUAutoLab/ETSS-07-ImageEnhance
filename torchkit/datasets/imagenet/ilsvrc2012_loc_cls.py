#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ImageNet dataset and datamodule.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageNet

from torchkit.core.data import ClassLabels
from torchkit.core.dataset import DataModule
from torchkit.core.image import show_images
from torchkit.core.runner import Phase
from torchkit.datasets.builder import DATAMODULES
from torchkit.utils import datasets_dir

__all__ = ["ILSVRC2012ClsDataModule"]


# MARK: - Data Config

data = {
	"name": "ilsvrc2012_cls",
	# The dataset's name.
	"shape": [256, 256, 3],
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


# MARK: - ILSVRC2012ClsDataModule

@DATAMODULES.register(name="ilsvrc2012_cls")
class ILSVRC2012ClsDataModule(DataModule):
	"""ImageNet 2012 Classification DataModule."""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		dataset_dir: str = os.path.join(datasets_dir, "imagenet", "ilsvrc2012"),
		name       : str = "imagenet",
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
		# NOTE: Load classlabels
		current_dir	    = os.path.dirname(os.path.abspath(__file__))
		classlabel_path = os.path.join(
			current_dir, "imagenet_classlabels.json"
		)
		self.classlabels = ClassLabels.create_from_file(
			label_path=classlabel_path
		)
	
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
			self.train = ImageNet(self.dataset_dir, split="train",
								  transform=self.transform)
			self.val   = ImageNet(self.dataset_dir, split="val",
								  transform=self.transform)
			
		# NOTE: Assign test datasets for use in dataloader(s)
		if phase in [None, Phase.TESTING]:
			self.val = ImageNet(self.dataset_dir, split="val",
								transform=self.transform)


# MARK: - Main

if __name__ == "__main__":
	# NOTE: Get DataModule
	cfg = data
	dm  = ILSVRC2012ClsDataModule(**cfg)
	dm.setup()
	# NOTE: Visualize labels
	# print(f"Classlabels \n{dm.classlabels.list}")
	# NOTE: Visualize an iteration
	data_iter = iter(dm.val_dataloader)
	data      = next(data_iter)
	print(data)
	images, targets = data
	show_images(images=images)
	plt.show(block=True)
