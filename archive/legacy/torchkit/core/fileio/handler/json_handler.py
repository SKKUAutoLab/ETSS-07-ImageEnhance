#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The handler for .json file.
"""

from __future__ import annotations

import json
from typing import Optional
from typing import TextIO
from typing import Union

import numpy as np

from .base import BaseFileHandler
from .builder import FILE_HANDLERS


# MARK: - JsonHandler

@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
	"""JSON file handler."""
	
	@staticmethod
	def set_default(obj):
		"""Set default json values for non-serializable values. It helps
		convert `set`, `range` and `np.ndarray` data types to list. It also
		converts `np.generic` (including `np.int32`, `np.float32`, etc.) into
		plain numbers of plain python built-in types.
		"""
		if isinstance(obj, (set, range)):
			return list(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.generic):
			return obj.item()
		raise TypeError(f"{type(obj)} is unsupported for json dump")
	
	def load_from_fileobj(
		self, path: Union[str, TextIO], **kwargs
	) -> Optional[Union[str, dict]]:
		"""Load the content from the given filepath or file-like object
		(input stream).
		"""
		return json.load(path)

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from the given obj to the filepath or file-like object.
		"""
		kwargs.setdefault("default", self.set_default)
		json.dump(obj, path, **kwargs)

	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from the given obj to string."""
		kwargs.setdefault("default", self.set_default)
		return json.dumps(obj, **kwargs)
