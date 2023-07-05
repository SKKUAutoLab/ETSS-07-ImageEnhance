#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes and Functions for reading and writing images and videos.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from ordered_enum import OrderedEnum
from PIL import ExifTags
from PIL.Image import Image

from torchkit.core.fileio import create_dirs
from torchkit.core.utils import Arrays
from torchkit.core.utils import Dim3
from torchkit.core.utils import ImageDict
from torchkit.core.utils import ImageList
from .imageproc import is_channel_first
from .imageproc import is_channel_last
from .imageproc import reshape_image
from .imageproc import to_channel_last
from .imageproc import unnormalize_image

logger = logging.getLogger()


# MARK: - ImageFormat

class ImageFormat(OrderedEnum):
	"""Define list of image format.
	"""
	
	BMP  = ".bmp"
	JPG  = ".jpg"
	JPEG = ".jpeg"
	PNG  = ".png"
	PPM  = ".ppm"
	
	@staticmethod
	def values():
		"""Return the list of all image formats.

		Returns:
			(list):
				The list of all image formats.
		"""
		return [e.value for e in ImageFormat]
	

# MARK: - VideoFormat

class VideoFormat(OrderedEnum):
	"""Define list of video format.
	"""
	AVI  = ".avi"
	M4V  = ".m4v"
	MKV  = ".mkv"
	MOV  = ".mov"
	MP4  = ".mp4"
	MPEG = ".mpeg"
	MPG  = ".mpg"
	WMV  = ".wmv"
	
	@staticmethod
	def values():
		"""Return the list of all video formats.

		Returns:
			(list):
				The list of all video formats.
		"""
		return [e.value for e in VideoFormat]


# MARK: - Validate

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
	if ExifTags.TAGS[orientation] == "Orientation":
		break


def is_image_file(path: str) -> bool:
	"""Check if the given path is an image file."""
	if path is None:
		return False
	
	image_formats = ImageFormat.values()  # Acceptable image suffixes
	if os.path.isfile(path=path):
		extension = os.path.splitext(path)[1]
		if extension in image_formats:
			return True
	return False


def is_video_file(path: Optional[str]) -> bool:
	"""Check if the given path is a video file."""
	if path is None:
		return False
	
	video_formats = VideoFormat.values()
	if os.path.isfile(path=path):
		extension = os.path.splitext(path)[1]
		if extension in video_formats:
			return True
	return False


def is_video_stream(path: str) -> bool:
	"""Check if the given path is a video stream."""
	return "rtsp" in path


# MARK: - Read

def exif_size(image: Image) -> tuple:
	"""Return the exif-corrected PIL size."""
	size = image.size  # (width, height)
	try:
		rotation = dict(image._getexif().items())[orientation]
		if rotation == 6:  # rotation 270
			size = (size[1], size[0])
		elif rotation == 8:  # rotation 90
			size = (size[1], size[0])
	except:
		pass
	return size[1], size[0]


# MARK: - Write

@dispatch(np.ndarray, str, str, str, str)
def write_image(
	image    : np.ndarray,
	dirpath  : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `PIL`.

	Args:
		image (np.ndarray):
			A single image.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		prefix (str):
			The filename prefix. Default: ``.
		extension (str):
			The image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if image.ndim not in [2, 3]:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{image.ndim}.")
	
	# NOTE: Unnormalize
	image = unnormalize_image(image)
	
	# NOTE: Convert to channel first
	if is_channel_first(image):
		image = reshape_image(image, False)
	
	# NOTE: Convert to PIL image
	if not Image.isImageType(t=image):
		image = Image.fromarray(image.astype(np.uint8))
	
	# NOTE: Write image
	if create_dirs(paths=[dirpath]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dirpath, f"{base}{extension}")
		else:
			filepath = os.path.join(dirpath, f"{prefix}_{base}{extension}")
		image.save(filepath)


@dispatch(torch.Tensor, str, str, str, str)
def write_image(
	image    : torch.Tensor,
	dirpath  : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `torchvision`.

	Args:
		image (torch.Tensor):
			A single image.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		prefix (str):
			The filename prefix. Default: ``.
		extension (str):
			The image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if image.dim() not in [2, 3]:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{image.dim()}.")
	
	# NOTE: Convert image
	image = unnormalize_image(image)
	image = reshape_image(image, True)
	
	# NOTE: Write image
	if create_dirs(paths=[dirpath]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dirpath, f"{base}{extension}")
		else:
			filepath = os.path.join(dirpath, f"{prefix}_{base}{extension}")
		
		if extension in [".jpg", ".jpeg"]:
			torchvision.io.image.write_jpeg(input=image, filename=filepath)
		elif extension in [".png"]:
			torchvision.io.image.write_png(input=image, filename=filepath)


@dispatch(np.ndarray, str, str, str)
def write_images(
	images   : np.ndarray,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `PIL`.

	Args:
		images (np.ndarray):
			A batch of images.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		extension (str):
			The image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if images.ndim != 4:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{images.ndim}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dirpath, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(torch.Tensor, str, str, str)
def write_images(
	images   : torch.Tensor,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `torchvision`.

	Args:
		images (torch.Tensor):
			A tensor of image.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		extension (str):
			The image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if images.dim() != 4:
		raise ValueError(f"Cannot save image with number of dimensions: "
						 f"{images.dim()}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dirpath, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(list, str, str, str)
def write_images(
	images   : ImageList,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (list):
			A list of images.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		extension (str):
			The image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, list) and
		all(isinstance(image, np.ndarray) for image in images)):
		cat_image = np.concatenate([images], axis=0)
		write_images(cat_image, dirpath, name, extension)
	elif (isinstance(images, list) and
		  all(torch.is_tensor(image) for image in images)):
		cat_image = torch.stack(images)
		write_images(cat_image, dirpath, name, extension)
	else:
		raise TypeError(f"Cannot concatenate images of type: {type(images)}.")


@dispatch(dict, str, str, str)
def write_images(
	images   : ImageDict,
	dirpath  : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (dict):
			A list of images.
		dirpath (str):
			The saving directory.
		name (str):
			Name of the image file.
		extension (str):
			The image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, dict) and
		all(isinstance(image, np.ndarray) for _, image in images.items())):
		cat_image = np.concatenate([image for key, image in images.items()],
								   axis=0)
		write_images(cat_image, dirpath, name, extension)
	elif (isinstance(images, dict) and
		  all(torch.is_tensor(image) for _, image in images)):
		values    = list(tuple(images.values()))
		cat_image = torch.stack(values)
		write_images(cat_image, dirpath, name, extension)
	else:
		raise TypeError


# MARK: - ImageLoader/Writer

class ImageLoader:
	"""Image Loader retrieves and loads image(s) from a filepath, a pathname
	pattern, or directory.

	Attributes:
		data (str):
			The data source. Can be a path to an image file or a directory.
			It can be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass.
		image_files (list):
			List of image files found in the data source.
		num_images (int):
			Total number of images.
		index (int):
			The current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data        = data
		self.batch_size  = batch_size
		self.image_files = []
		self.num_images  = -1
		self.index       = 0
		
		self.init_image_files(data=self.data)

	def __len__(self):
		"""Return the number of images in the `image_files`."""
		return self.num_images  # Number of images
	
	def __iter__(self):
		"""Return an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""The next iterator.
		
		Examples:
			>>> video_stream = ImageLoader("cam_1.mp4")
			>>> for index, image in enumerate(video_stream):
		
		Returns:
			images (np.ndarray):
				The list of image file from opencv with `np.ndarray` type.
			indexes (list):
				The list of image indexes.
			files (list):
				The list of image files.
			rel_paths (list):
				The list of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_images:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_images:
					break
				
				file     = self.image_files[self.index]
				rel_path = file.replace(self.data, "")

				images.append(cv2.imread(self.image_files[self.index]))
				indexes.append(self.index)
				files.append(file)
				rel_paths.append(rel_path)

				self.index += 1

			return np.array(images), indexes, files, rel_paths
	
	# MARK: Configure
	
	def init_image_files(self, data: str):
		"""Initialize list of image files in data source.
		
		Args:
			data (str):
				The data source. Can be a path to an image file or a directory.
				It can be a pathname pattern to images.
		"""
		if is_image_file(data):
			self.image_files = [data]
		elif os.path.isdir(data):
			self.image_files = [
				img for img in glob(os.path.join(data, "**/*"), recursive=True)
				if is_image_file(img)
			]
		elif isinstance(data, str):
			self.image_files = [img for img in glob(data) if is_image_file(img)]
		else:
			raise IOError("Error when reading input image files.")
		self.num_images = len(self.image_files)

	def list_image_files(self, data: str):
		"""Alias of `init_image_files()`."""
		self.init_image_files(data=data)


class ImageWriter:
	"""Video Writer saves images to a destination directory.

	Attributes:
		dst (str):
			The output directory or filepath.
		extension (str):
			The image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
		index (int):
			The current index. Default: `0`.
	"""

	# MARK: Magic Functions

	def __init__(self, dst: str, extension: str = ".jpg"):
		super().__init__()
		self.dst	   = dst
		self.extension = extension
		self.index     = 0

	def __len__(self):
		"""Return the number of already written images."""
		return self.index

	# MARK: Write

	def write_image(self, image: np.ndarray, image_file: Optional[str] = None):
		"""Write image.

		Args:
			image (np.ndarray):
				The image.
			image_file (str, optional):
				The image file.
		"""
		if is_channel_last(image):
			image = to_channel_last(image)
		
		if image_file is not None:
			image_file = (image_file[1:] if image_file.startswith("\\")
						  else image_file)
			image_name = os.path.splitext(image_file)[0]
		else:
			image_name = f"{self.index}"
		
		output_file = os.path.join(self.dst, f"{image_name}{self.extension}")
		parent_dir  = str(Path(output_file).parent)
		create_dirs(paths=[parent_dir])
		
		cv2.imwrite(output_file, image)
		self.index += 1

	def write_images(
		self, images: Arrays, image_files: Optional[list[str]] = None
	):
		"""Write batch of images.

		Args:
			images (Arrays):
				The images.
			image_files (list[str], optional):
				The image files.
		"""
		if image_files is None:
			image_files = [None for _ in range(len(images))]
		# assert len(image_files) == len(images), \
		# 	f"{len(image_files)} != {len(images)}"

		for image, image_file in zip(images, image_files):
			self.write_image(image=image, image_file=image_file)


# MARK: - VideoLoader/Writer

class VideoLoader:
	"""Video Loader loads frames from a video file or a video stream.

	Attributes:
		data (str):
			The data source. Can be a path to video file or a stream link.
		batch_size (int):
			Number of samples in one forward & backward pass.
		video_capture (VideoCapture):
			The `VideoCapture` object from OpenCV.
		num_frames (int):
			Total number of frames in the video.
		index (int, optional):
			The current frame index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data          = data
		self.batch_size    = batch_size
		self.video_capture = None
		self.num_frames    = -1
		self.index         = 0

		self.init_video_capture(data=self.data)
		
	def __len__(self):
		"""Return the number of frames in the video.

		Returns:
			num_frames (int):
				>0 if the offline video.
				-1 if the online video.
		"""
		return self.num_frames  # number of frame, [>0 : video, -1 : online_stream]

	def __iter__(self):
		"""Returns an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""
		e.g.:
				>>> video_stream = VideoLoader("cam_1.mp4")
				>>> for image, index in enumerate(video_stream):

		Returns:
			images (np.ndarray):
				The list of numpy.array images from OpenCV.
			indexes (list):
				The list of image indexes in the video.
			files (list):
				The list of image files.
			rel_paths (list):
				The list of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_frames:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_frames:
					break
					
				ret_val, image = self.video_capture.read()
				rel_path       = os.path.basename(self.data)
				
				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)
				
				self.index += 1

			return np.array(images), indexes, files, rel_paths

	def __del__(self):
		"""Close the `video_capture` object."""
		self.close()

	# MARK: Configure
	
	def init_video_capture(self, data: str):
		"""Initialize `video_capture` object.
		
		Args:
			data (str):
				The data source. Can be a path to video file or a stream link.
		"""
		if is_video_file(data):
			self.video_capture = cv2.VideoCapture(data)
			self.num_frames    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif is_video_stream(data):
			self.video_capture = cv2.VideoCapture(data)  # stream
			# Set buffer (batch) size
			self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)
		
		if self.video_capture is None:
			raise IOError("Error when reading input stream or video file!")

	def close(self):
		"""Release the current `video_capture` object."""
		if self.video_capture:
			self.video_capture.release()


class VideoWriter:
	"""Video Writer saves images to a video file.

	Attributes:
		dst (str):
			The output video file.
		video_writer (VideoWriter):
			The `VideoWriter` object from OpenCV.
		shape (tuple):
			Output size as [H, W, C]. This is also used to reshape the input.
		frame_rate (int):
			The frame rate of the video.
		fourcc (str):
			The video codec. One of: ["mp4v", "xvid", "mjpg", "wmv1"].
		save_image (bool):
			Should write individual image?
		index (int):
			The current index.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		dst       : str,
		shape     : Dim3  = (480, 640, 3),
		frame_rate: float = 10,
		fourcc    : str   = "mp4v",
		save_image: bool  = False,
	):
		super().__init__()
		self.shape        = shape
		self.frame_rate   = frame_rate
		self.fourcc       = fourcc
		self.save_image	  = save_image
		self.video_writer = None
		self.index		  = 0

		self.init_video_writer(dst=dst)

	def __len__(self):
		"""Return the number of already written frames."""
		return self.index

	def __del__(self):
		"""Close the `video_writer` object."""
		self.close()

	# MARK: Configure
	
	def init_video_writer(self, dst: str):
		"""Initialize `video_writer` object.

		Args:
			dst (str):
				The output video file.
		"""
		if os.path.isdir(dst):
			parent_dir = dst
			self.dst   = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(dst).parent)
			stem       = str(Path(dst).stem)
			self.dst   = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])

		fourcc            = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			self.dst, fourcc, self.frame_rate,
			tuple([self.shape[1], self.shape[0]])  # Must be [W, H]
		)

		if self.video_writer is None:
			raise FileNotFoundError(f"Video file cannot be created at "
									f"{self.dst}.")

	def close(self):
		"""Release the `video_writer` object."""
		if self.video_writer:
			self.video_writer.release()

	# MARK: Write

	def write_frame(self, image: np.ndarray):
		"""Add a frame to video.

		Args:
			image (np.ndarray):
				The image for writing of shape [H, W, C].
		"""
		if is_channel_last(image):
			image = to_channel_last(image)

		if self.save_image:
			parent_dir = os.path.splitext(self.dst)[0]
			image_file = os.path.join(parent_dir, f"{self.index}.png")
			create_dirs(paths=[parent_dir])
			cv2.imwrite(image_file, image)

		self.video_writer.write(image)
		self.index += 1

	def write_frames(self, images: Arrays):
		"""Add batch of frames to video.

		Args:
			images (Arrays):
				The images.
		"""
		for image in images:
			self.write_frame(image=image)


# MARK: - FrameLoader/Writer

class FrameLoader:
	"""Frame Loader retrieves and loads frame(s) from a filepath, a pathname
	pattern, a directory, a video, or a stream.

	Attributes:
		data (str):
			The data source. Can be a path to an image file, a directory,
			a video, or a stream. It can also be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass.
		image_files (list):
			List of image files found in the data source.
		video_capture (VideoCapture):
			The VideoCapture object from OpenCV.
		num_frames (int):
			Total number of image files or total number of frames in the video.
		index (int):
			The current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data          = data
		self.batch_size    = batch_size
		self.image_files   = []
		self.video_capture = None
		self.num_frames    = -1
		self.index         = 0

		self.init_image_files_or_video_capture(data=self.data)

	def __len__(self):
		"""Get the number of frames in the video or the number of images in
		`image_files`.

		Returns:
			num_frames (int):
				>0 if the offline video.
				-1 if the online video.
		"""
		return self.num_frames  # number of frame, [>0 : video, -1 : online_stream]

	def __iter__(self):
		"""Return an iterator starting at index 0.

		Returns:
			self (VideoInputStream):
				For definition __next__ below.
		"""
		self.index = 0
		return self

	def __next__(self):
		"""The next items.
			e.g.:
				>>> video_stream = VideoLoader("cam_1.mp4")
				>>> for image, index in enumerate(video_stream):

		Returns:
			images (np.ndarray):
				The list of image file from opencv with `np.ndarray` type.
			indexes (list):
				The list of image indexes.
			files (list):
				The list of image files.
			rel_paths (list):
				The list of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_frames:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_frames:
					break

				if self.video_capture:
					ret_val, image = self.video_capture.read()
					rel_path 	   = os.path.basename(self.data)
				else:
					image	 = cv2.imread(self.image_files[self.index])
					file     = self.image_files[self.index]
					rel_path = file.replace(self.data, "")

				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)

				self.index += 1

			return np.array(images), indexes, files, rel_paths

	def __del__(self):
		"""Close `video_capture` object."""
		self.close()

	# MARK: Configure

	def init_image_files_or_video_capture(self, data: str):
		"""Initialize image files or `video_capture` object.

		Args:
			data (str):
				The data source. Can be a path to an image file, a directory,
				a video, or a stream. It can also be a pathname pattern to
				images.
		"""
		if is_video_file(data):
			self.video_capture = cv2.VideoCapture(data)
			self.num_frames    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif is_video_stream(data):
			self.video_capture = cv2.VideoCapture(data)  # stream
			self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)  # set buffer (batch) size
			self.num_frames    = -1
		elif is_image_file(data):
			self.image_files = [data]
			self.num_frames  = len(self.image_files)
		elif os.path.isdir(data):
			self.image_files = [img for img in glob.glob(os.path.join(data, "**/*"), recursive=True) if is_image_file(img)]
			self.num_frames  = len(self.image_files)
		elif isinstance(data, str):
			self.image_files = [img for img in glob.glob(data) if is_image_file(img)]
			self.num_frames  = len(self.image_files)
		else:
			raise IOError(f"Error when reading data!")

	def close(self):
		"""Release the `video_capture` object."""
		if self.video_capture:
			self.video_capture.release()


class FrameWriter:
	"""Frame Writer saves frames to individual image files or appends all to a
	video file.

	Attributes:
		dst (str):
			The output video file or a directory.
		video_writer (VideoWriter):
			The `VideoWriter` object from OpenCV.
		shape (tuple):
			Output size as [H, W, C]. This is also used to reshape the input.
		frame_rate (int):
			The frame rate of the video.
		fourcc (str):
			The video codec. One of: ["mp4v", "xvid", "mjpg", "wmv1"].
		save_image (bool):
			Should write individual image?
		save_video (bool):
			Should write video?
		index (int):
			The current index.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		dst		  : str,
		shape     : Dim3  = (480, 640, 3),
		frame_rate: float = 10,
		fourcc    : str   = "mp4v",
		save_image: bool  = False,
		save_video: bool  = True,
	):
		"""

		Args:
			dst (str):
				The output video file or a directory.
			shape (tuple):
				Output size as [H, W, C]. This is also used to reshape the
				input.
			frame_rate (int):
				The frame rate of the video.
			fourcc (str):
				The video codec. One of: ["mp4v", "xvid", "mjpg", "wmv1"].
			save_image (bool):
				Should write individual image?
			save_video (bool):
				Should write video?
		"""
		super().__init__()
		self.dst		  = dst
		self.shape        = shape
		self.frame_rate   = frame_rate
		self.fourcc       = fourcc
		self.save_image   = save_image
		self.save_video   = save_video
		self.video_writer = None
		self.index		  = 0

		if self.save_video:
			self.init_video_writer()

	def __len__(self):
		"""Return the number of already written frames."""
		return self.index

	def __del__(self):
		"""Close the `video_writer`."""
		self.close()

	# MARK: Configure

	def init_video_writer(self):
		"""Initialize `video_writer` object."""
		if os.path.isdir(self.dst):
			parent_dir = self.dst
			video_file = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(self.dst).parent)
			stem       = str(Path(self.dst).stem)
			video_file = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])

		fourcc			  = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			video_file, fourcc, self.frame_rate,
			tuple([self.shape[1], self.shape[0]])  # Must be [W, H]
		)

		if self.video_writer is None:
			raise FileNotFoundError(f"Video file cannot be created at "
									f"{video_file}.")

	def close(self):
		"""Close the `video_writer`."""
		if self.video_writer:
			self.video_writer.release()

	# MARK: Write

	def write_frame(self, image: np.ndarray, image_file: Optional[str] = None):
		"""Add a frame to writing video.

		Args:
			image (np.ndarray):
				The image for writing of shape [H, W, C].
			image_file (str, optional):
				The image file. Default: `None`.
		"""
		if is_channel_last(image):
			image = to_channel_last(image)

		if self.save_image:
			if image_file is not None:
				image_file = (image_file[1:] if image_file.startswith("\\")
							  else image_file)
				image_name = os.path.splitext(image_file)[0]
			else:
				image_name = f"{self.index}"
			output_file = os.path.join(self.dst, f"{image_name}.png")
			parent_dir  = str(Path(output_file).parent)
			create_dirs(paths=[parent_dir])
			cv2.imwrite(output_file, image)
		if self.save_video:
			self.video_writer.write(image)

		self.index += 1

	def write_frames(
		self, images: Arrays, image_files: Optional[list[str]] = None
	):
		"""Add batch of frames to video.

		Args:
			images (Arrays):
				The images.
			image_files (list[str], optional):
				The image files. Default: `None`.
		"""
		if image_files is None:
			image_files = [None for _ in range(len(images))]

		for image, image_file in zip(images, image_files):
			self.write_frame(image=image, image_file=image_file)
