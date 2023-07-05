# ==================================================================== #
# File name: replicate_no_rain_image.py
# Author: Long H. Pham
# Date created: 09/25/2021
# The `torchkit.datasets.rain.utils.replicate_no_rain_image`
# ==================================================================== #
from __future__ import annotations

import glob
import os
import shutil

import cv2
from tqdm import tqdm

from torchkit.core.fileio import create_dirs
from torchkit.utils import datasets_dir


image_dir             = os.path.join(datasets_dir, "rain", "rain1400", "test")
rain_dir              = os.path.join(image_dir, "rain")
no_rain_dir           = os.path.join(image_dir, "no_rain")
no_rain2_dir           = os.path.join(image_dir, "no_rain2")
rain_image_pattern    = os.path.join(rain_dir, "*.jpg")
no_rain_image_pattern = os.path.join(no_rain_dir, "*.jpg")

create_dirs(paths=[os.path.join(image_dir, "no_rain2")])

# NOTE: Read images
for path in tqdm(glob.glob(rain_image_pattern)):
	rain_image         = cv2.imread(path)
	rain_filename      = os.path.basename(path)
	no_rain_filename   = rain_filename[:rain_filename.find("_")] + ".jpg"
	no_rain_path       = os.path.join(no_rain_dir, no_rain_filename)
	shutil.copy(no_rain_path, os.path.join(no_rain2_dir, rain_filename))
	#no_rain_image      = cv2.imread(no_rain_path)
	#cv2.imwrite(path.replace("rain", "no_rain2"), no_rain_image)
