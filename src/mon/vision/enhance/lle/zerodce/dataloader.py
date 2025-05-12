import glob
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)


def populate_train_list(image_dir: str):
	image_list = glob.glob(image_dir + "*.jpg")
	random.shuffle(image_list)
	return image_list


class LowLightLoader(data.Dataset):
	
	def __init__(self, image_dir: str):
		self.size       = 256
		self.train_list = populate_train_list(image_dir)
		self.data_list  = self.train_list
		# print("Total training examples:", len(self.train_list))

	def __getitem__(self, index: int) -> torch.Tensor:
		image = Image.open(self.data_list[index])
		image = image.resize((self.size, self.size), Image.ANTIALIAS)
		image = (np.asarray(image) / 255.0)
		image = torch.from_numpy(image).float()
		image = image.permute(2, 0, 1)
		return image

	def __len__(self) -> int:
		return len(self.data_list)
