#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends Python ``pathlib`` module."""

__all__ = [
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
    "copy_file",
    "delete_cache",
    "delete_dir",
    "delete_files",
    "hash_files",
    "mkdirs",
    "parse_data_dir",
    "parse_output_dir",
    "parse_save_dir",
    "rmdirs",
]

import os
import pathlib
import shutil
from pathlib import *

import validators

from mon.core import humps, type_extensions


# ----- Path Class -----
class Path(type(pathlib.Path())):
    """Extended ``pathlib.Path`` with additional functionalities.
    
    Notes:
        Methods are kept as methods (not properties) for consistency with ``pathlib.Path``.
    """
    
    # ----- Check Internal Parts -----
    def is_basename(self) -> bool:
        """Checks if the path is a file basename.

        Returns:
            ``True`` if path matches its basename, ``False`` otherwise.
        """
        return str(self) == self.name
     
    def is_name(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_stem(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_url(self) -> bool:
        """Checks if the path is a valid URL.

        Returns:
            ``True`` if path is a valid URL, ``False`` otherwise.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Checks if the path is a file or valid URL.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a file or valid URL, ``False`` otherwise.
        """
        return ((not exist or self.is_file()) or
                not isinstance(validators.url(str(self)), validators.ValidationError))
    
    def is_file_like(self) -> bool:
        """"Checks if the path resembles a file format.

        Returns:
            ``True`` if path has a suffix, ``False`` otherwise.
        """
        return "." in self.suffix
    
    def is_dir_like(self) -> bool:
        """Checks if the path resembles a directory format.

        Returns:
            ``True`` if path has no suffix, ``False`` otherwise.
        """
        return self.suffix == ""
    
    def has_subdir(self, name: str) -> bool:
        """Checks if the directory has a subdirectory with the given name.

        Args:
            name: Subdirectory name to check.

        Returns:
            ``True`` if subdirectory exists, ``False`` otherwise.
        """
        return name in [d.name for d in self.subdirs()]
    
    # ----- Check Text File -----
    def is_json_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.json`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.json`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.txt`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.txt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Checks if the path is an ``.xml`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is an ``.xml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.yaml`` or ``.yml`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.yaml`` or ``.yml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]
   
    # ----- Check Image File -----
    def is_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is an image file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is an image file, ``False`` otherwise.
        """
        from mon.constants import ImageExtension
        return (not exist or self.is_file()) and self.suffix.lower() in ImageExtension
        
    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is a raw image file (``.dng`` or ``.arw``).

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a raw image file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]
    
    # ----- Check Video File -----
    def is_video_file(self, exist: bool = True) -> bool:
        """Checks if the path is a video file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a video file, ``False`` otherwise.
        """
        from mon.constants import VideoExtension
        return (not exist or self.is_file()) and self.suffix.lower() in VideoExtension
    
    def is_video_stream(self) -> bool:
        """Checks if the path is a video stream.

        Returns:
            ``True`` if path contains ``rtsp``, ``False`` otherwise.
        """
        return "rtsp" in str(self).lower()
    
    # ----- Check Torch File -----
    def is_cache_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.cache`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.cache`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.ckpt`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.ckpt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"
    
    def is_config_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.config`` or ``.cfg`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a config file, ``False`` otherwise.
        """
        from mon.constants import ConfigExtension
        return (not exist or self.is_file()) and self.suffix.lower() in ConfigExtension
        
    def is_py_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.py`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a ``.py`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"
    
    def is_torch_file(self, exist: bool = True) -> bool:
        """Checks if the path is a Torch-compatible file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path has a Torch extension, ``False`` otherwise.
        """
        from mon.constants import TorchExtension
        return (not exist or self.is_file()) and self.suffix.lower() in TorchExtension
    
    def is_weights_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.pt`` or ``.pth`` file.

        Args:
            exist: If ``True``, verifies file exists. Default is ``True``.

        Returns:
            ``True`` if path is a weights file, ``False`` otherwise.
        """
        from mon.constants import WeightExtension
        return (not exist or self.is_file()) and self.suffix.lower() in WeightExtension
    
    # ----- List -----
    def subdirs(self, recursive: bool = False) -> list["Path"]:
        """Returns a list of subdirectory paths.

        Args:
            recursive: If ``True``, includes subdirs recursively. Default is ``False``.

        Returns:
            List of subdirectory ``Path`` objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]
    
    def files(self, recursive: bool = False) -> list["Path"]:
        """Returns a list of file paths in the directory.

        Args:
            recursive: If ``True``, includes files in subdirs. Default is ``False``.

        Returns:
            List of file ``Path`` objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]
    
    def ckpt_file(self) -> "Path":
        """Returns the checkpoint file path if found.

        Returns:
            Checkpoint file ``Path`` or ``None`` if not found.
        """
        ckpt_path = self.with_suffix(".ckpt")
        return ckpt_path if ckpt_path.is_file() else self
    
    def config_file(self) -> "Path":
        """Returns the configuration file path.

        Returns:
            Configuration file ``Path``.
        """
        from mon.constants import ConfigExtension
        for ext in ConfigExtension.values():
            for stem in [self.stem, humps.snakecase(self.stem)]:
                config_path = self.with_name(f"{stem}{ext}")
                if config_path.is_file():
                    return config_path
        return self
    
    def latest_file(self) -> "Path":
        """Returns the latest file based on creation time.

        Returns:
            Latest file ``Path`` or ``None`` if no files exist.
        """
        files = self.files()
        return max(files, key=os.path.getctime) if files else None
    
    def image_file(self) -> "Path":
        """Returns the image file path.

        Returns:
            Image file ``Path``.
        """
        from mon.constants import ImageExtension
        for ext in ImageExtension.values():
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def yaml_file(self) -> "Path":
        """Returns the YAML file path.

        Returns:
            YAML file ``Path``.
        """
        for ext in [".yaml", ".yml"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def relative_path(self, start_part: str) -> "Path":
        """Returns the relative path from a given start part.

        Args:
            start_part: Starting path or string for relativity.

        Returns:
            Relative ``Path`` from ``start_part``.
        """
        path       = Path(self)
        start_part = str(start_part)
        path_str   = str(path)
        if start_part not in path_str:
            return path
        start_idx = path_str.index(start_part)
        return Path(path_str[start_idx:])
    
    # ----- Creation -----
    def copy_to(self, dst: str, replace: bool = True):
        """Copies the file to a new location.

        Args:
            dst: Destination path or string.
            replace: If ``True``, replaces existing file. Default is ``True``.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError("[dst] as a URL is not supported.")
        mkdirs(dst, parents=True, exist_ok=True)
        dst = dst / self.name if dst.is_dir_like() else dst
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))
    
    def replace(self, old: str, new: str, count: int = 1) -> "Path":
        """Replaces occurrences of a string in the path.

        Args:
            old: String to replace.
            new: Replacement string.
            count: Max number of replacements. Default is ``1``.

        Returns:
            New ``Path`` with replaced string.
        """
        return Path(str(self).replace(old, new, count))


# ----- Create -----
def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copies a file to a new location.

    Args:
        src: Path to the source file.
        dst: Path to the destination.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ----- Read -----
def hash_files(paths: list[Path | str]) -> int:
    """Calculates the total hash value of files based on their sizes.

    Args:
        paths: List of file paths to hash.

    Returns:
        Integer sum of file sizes in bytes.
    """
    paths = [Path(f) for f in type_extensions.to_list(paths) if f]
    return sum(f.stat().st_size for f in paths if f.is_file())


# ----- Delete -----
def delete_cache(path: Path | str, recursive: bool = True):
    """Clears cache files in a directory and optionally its subdirs.

    Args:
        path: Directory path containing cache files.
        recursive: If ``True``, searches subdirs. Default is ``True``.
    """
    delete_files(path=path, regex=".cache", recursive=recursive)


def delete_dir(paths: Path | str | list[Path | str]):
    """Deletes directories and their contents.

    Args:
        paths: Single path or list of directory paths.
    """
    paths = type_extensions.unique(type_extensions.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.exists():
            delete_files(path=p, regex="*", recursive=True)
            shutil.rmtree(p)


def delete_files(
    path     : Path | str,
    regex    : str  = None,
    recursive: bool = False
):
    """Deletes files matching a pattern in a directory.

    Args:
        path: Directory path to search for files.
        regex: File path pattern. Default is ``None`` (deletes ``path`` if file).
        recursive: If ``True``, searches subdirs. Default is ``False``.
    """
    path = Path(path)
    if regex:
        path  = path.parent if not path.is_dir() else path
        files = list(path.rglob(regex)) if recursive else list(path.glob(regex))
    else:
        files = [path]
    for f in files:
        try:
            f.unlink()
        except Exception as err:
            print(f"Cannot delete file: [err].")


def mkdirs(
    paths   : Path | str | list[Path | str],
    mode    : int  = 0o777,
    parents : bool = True,
    exist_ok: bool = True,
    replace : bool = False,
):
    """Creates directories with specified options.

    Args:
        paths: Single path or list of directory paths.
        mode: File mode with umask. Default is ``0o777``.
        parents: If ``True``, creates parents. Default is ``True``.
        exist_ok: If ``True``, ignores existing dirs. Default is ``True``.
        replace: If ``True``, deletes and recreates dirs. Default is ``False``.
    """
    paths = type_extensions.unique(type_extensions.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.is_url():
            continue
        p = p.parent if p.is_file_like() else p
        if replace and p.exists():
            delete_files(path=p, regex="*")
            p.rmdir()
        p.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def rmdirs(paths: Path | str | list[Path | str]):
    """Deletes directories and their contents.

    Args:
        paths: Single path or list of directory paths.
    """
    paths = type_extensions.unique(type_extensions.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.is_url():
            continue
        if p.is_file_like():
            p = p.parent
        delete_files(path=p, regex="*")
        try:
            p.rmdir()
        except Exception as err:
            print(f"Cannot delete directory: [err].")


# ----- Convert -----
def parse_data_dir(root: str | pathlib.Path, data_dir: str | pathlib.Path = "") -> str | pathlib.Path:
    """Parses the absolute data directory path from given components.

    Args:
        root: Root directory.
        data_dir: Data directory.

    Returns:
        Parsed the absolute path of the data directory.
    """
    from mon.constants import ROOT_DIR
    
    root_      = pathlib.Path(root)     if root     not in [None, "None", ""] else ROOT_DIR
    data_dir_  = pathlib.Path(data_dir) if data_dir not in [None, "None", ""] else None

    candidates = []
    if data_dir_:
        candidates.extend([
            data_dir_,
            root_    / data_dir_,
            root_    / "data" / data_dir_,
            ROOT_DIR / data_dir_,
            ROOT_DIR / "data" / data_dir_
        ])
    candidates.extend([
        root_    / "data",
        ROOT_DIR / "data"
    ])

    for d in candidates:
        if d.is_dir():
            return d

    return root


def parse_save_dir(
    root : str | Path,
    arch : str = None,
    model: str = None,
    data : str = None,
) -> str | pathlib.Path:
    """Parses a save dir in format: root/arch/model/data.

    Args:
        root: Project root.
        arch: Model architecture. Default is ``None``.
        model: Model name. Default is ``None``.
        data: Dataset name. Default is ``None``.

    Returns:
        Parsed save dir path as ``str`` or ``pathlib.Path``.
    """
    save_dir = Path(root)
    data     = Path(data) if data not in [None, "None", ""] else None
    if arch:
        save_dir /= arch
    if model:
        save_dir /= model
        if data:
            if data.is_dir() or data.is_file():
                save_dir /= data.stem
            else:
                save_dir /= data
    return save_dir


def parse_output_dir(
    root        : str | Path,
    dirname     : str | Path,
    subdir_name : str | Path,
    src_path    : str | Path,
    keep_subdirs: bool = False,
    save_nearby : bool = False,
) -> pathlib.Path:
    """Parses the output directory path from given components.

    It should be in this pattern:
        ``root/dirname/file``.
        where:
            root = ``parse_save_dir()``
    
    Args:
        root: Root directory. Root is assumed to be in this pattern: ``root/arch/model/[fullname or dirname]``.
        src_path: Source file path.
        dirname: Directory name.
        subdir_name: Subdirectory name. Default is ``None``.
        keep_subdirs: If ``True``, keeps subdirectories in the path.
            Default is ``False``.
        save_nearby: If ``True``, saves in the same parent directory as the file.
            Default is ``False``.

    Example:
        root     = ../enhance/run/predict/zerodce/zerodce/dicm
        dirname  = dicm
        src_path = ../enhance/data/dicm/test/image/0001.jpg
        rel_path = dicm/test/image/0001.jpg
        return   : ../enhance/run/predict/zerodce/zerodce/dicm/test/image
    """
    root        = Path(root)
    dirname     = Path(dirname)
    subdir_name = subdir_name if subdir_name not in [None, "None", ""] else None
    subdir_name = None if save_nearby else subdir_name
    src_path    = Path(src_path)

    # Update root and dirname
    if save_nearby:
        if root.stem == dirname.stem:
            root_suffix = root.parent.stem
        else:
            root_suffix = root.stem
        root    = src_path.parent.parent / f"{src_path.parent.stem}_{root_suffix}"
        dirname = Path(src_path.parent.stem)

    if keep_subdirs:
        rel_path = src_path.relative_path(dirname)
        if subdir_name:
            return root / subdir_name / rel_path.parent
        else:
            return root / rel_path.parent
    else:
        if not save_nearby and dirname.stem != root.stem:
            root = root / dirname.stem
        if subdir_name:
            return root / subdir_name
        else:
            return root
