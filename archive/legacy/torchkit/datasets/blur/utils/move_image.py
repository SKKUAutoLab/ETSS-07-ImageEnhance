#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
import shutil

from tqdm import tqdm

from torchkit.utils import datasets_dir


image_dir          = os.path.join(datasets_dir, "blur", "hide", "test", "near")
gt_dir             = os.path.join(datasets_dir, "blur", "hide", "GT")
blur_dir           = os.path.join(image_dir, "blur")
no_blur_dir        = os.path.join(image_dir, "no_blur")
blur_image_pattern = os.path.join(blur_dir, "*.png")

# NOTE: Read images
for path in tqdm(glob.glob(blur_image_pattern)):
	blur_filename = os.path.basename(path)
	no_blur_path  = os.path.join(gt_dir, blur_filename)
	shutil.copy(no_blur_path, os.path.join(no_blur_dir, blur_filename))
