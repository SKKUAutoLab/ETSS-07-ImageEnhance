#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
import os

from torch import distributed as dist

logger = logging.getLogger()


def get_dist_info():
	if dist.is_available():
		initialized = dist.is_initialized()
	else:
		initialized = False
	if initialized:
		rank       = dist.get_rank()
		world_size = dist.get_world_size()
	else:
		rank       = 0
		world_size = 1
	return rank, world_size


# noinspection PyTypeChecker
def get_next_version(root_dir: str) -> int:
	"""Get the next experiment version number.
	
	Args:
		root_dir (str):
			The path to the folder that contains all experiment folders.

	Returns:
		version (int):
			The next version number.
	"""
	try:
		listdir_info = os.listdir(root_dir)
	except OSError:
		logger.warning("Missing folder: %s", root_dir)
		return 0
	
	existing_versions = []
	for listing in listdir_info:
		if isinstance(listing, str):
			d  = listing
		else:
			d  = listing["name"]
		bn = os.path.basename(d)
		if bn.startswith("version_"):
			dir_ver = bn.split("_")[1].replace("/", "")
			existing_versions.append(int(dir_ver))
	if len(existing_versions) == 0:
		return 0
	
	return max(existing_versions) + 1
