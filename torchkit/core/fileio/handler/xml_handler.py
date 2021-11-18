# ==================================================================== #
# File name: xml_handler.py
# Author: Long H. Pham
# Date created: 07/19/2021
# The `torchkit.core.fileio.handler.xml_handler` implements the file
# handler for .xml file.
# ==================================================================== #
from __future__ import annotations

from typing import TextIO
from typing import Union

import xmltodict

from .base import BaseFileHandler
from .builder import FILE_HANDLERS


# MARK: - XmlHandler

@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
	"""XML file handler.
	"""
	
	def load_from_fileobj(self, path: Union[str, TextIO], **kwargs) -> Union[str, dict, None]:
		"""Load data from file object (input stream).

		Args:
			path (str, TextIO):
				The filepath or a file-like object.

		Returns:
			(str, dict, optional):
				The content from the file.
		"""
		with open(path) as f:
			doc = xmltodict.parse(f.read())
		return doc

	def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
		"""Dump data from obj to file.

		Args:
			obj:
				The object.
			path (str, TextIO):
				The filepath or a file-like object.
		"""
		assert isinstance(obj, dict), f"obj {obj} must be a dictionary."
		with open(path, "w") as path:
			path.write(xmltodict.unparse(obj, pretty=True))
		
	def dump_to_str(self, obj, **kwargs) -> str:
		"""Dump data from obj to string.

		Args:
			obj:
				The object.

		Returns:
			(str):
				The content from the file.
		"""
		assert isinstance(obj, dict), f"obj {obj} must be a dictionary."
		return xmltodict.unparse(obj, pretty=True)
