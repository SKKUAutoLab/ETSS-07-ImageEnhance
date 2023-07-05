# ==================================================================== #
# File name: split_image.py
# Author: Long H. Pham
# Date created: 09/25/2021
# The `torchkit.datasets.rain.utils.split_image` implements the utility
# functions to split the given image into rain and no-rain images.
# ==================================================================== #
from __future__ import annotations

import glob
import os

import cv2
from tqdm import tqdm

from torchkit.core.fileio import create_dirs
from torchkit.utils import datasets_dir


image_dir     = os.path.join(datasets_dir, "rain", "rain1200", "train", "medium")
image_pattern = os.path.join(image_dir, "*.jpg")
rain_dir      = os.path.join(image_dir, "rain")
no_rain_dir   = os.path.join(image_dir, "no_rain")

create_dirs(paths=[rain_dir, no_rain_dir])

# NOTE: Read images
image_paths = glob.glob(image_pattern)
for path in tqdm(image_paths):
	image         = cv2.imread(path)
	h, w, c       = image.shape
	no_rain_image = image[:, 0 : int(w / 2), :]
	rain_image    = image[:, int(w / 2) : w, :]
	
	filename           = os.path.basename(path)
	rain_image_path    = os.path.join(rain_dir, filename)
	no_rain_image_path = os.path.join(no_rain_dir, filename)
	cv2.imwrite(rain_image_path, rain_image)
	cv2.imwrite(no_rain_image_path, no_rain_image)
