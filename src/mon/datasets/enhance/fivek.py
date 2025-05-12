#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MIT-Adobe FiveK datasets."""

__all__ = [
    "FiveK",
    "FiveKA",
    "FiveKADataModule",
    "FiveKB",
    "FiveKBDataModule",
    "FiveKC",
    "FiveKCDataModule",
    "FiveKD",
    "FiveKDDataModule",
    "FiveKDataModule",
    "FiveKE",
    "FiveKEDataModule",
    "FiveKInit",
    "FiveKInitDataModule",
]

from collections import defaultdict
from typing import Literal

import numpy as np
import torch

from mon import core, vision
from mon.constants import DATAMODULES, DATASETS, Split, Task

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
InfraredAnnotation             = vision.InfraredAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="fivekinit")
class FiveKInit(VisionDataset):
    """Loads FiveKInit dataset from ``root`` dir for model init.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE, Task.RETOUCH]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image_ex": ImageAnnotation,
        "image_bc": ImageAnnotation,
        "image_vb": ImageAnnotation,
        "ref_ex"  : ImageAnnotation,
        "ref_bc"  : ImageAnnotation,
        "ref_vb"  : ImageAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        self.file_ex   = defaultdict(list)
        self.file_bc   = defaultdict(list)
        self.file_vb   = defaultdict(list)
        self.file_keys = []
        super().__init__(root=root, *args, **kwargs)

    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at given ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            ``dict`` with datapoint and metadata.
        """
        key = self.file_keys[index]
        A_ex, B_ex = np.random.choice(self.file_ex[key], 2, replace=False)
        A_bc, B_bc = np.random.choice(self.file_bc[key], 2, replace=False)
        A_vb, B_vb = np.random.choice(self.file_vb[key], 2, replace=False)

        val_ex = torch.tensor((int(B_ex.stem.split("-")[-1]) - int(A_ex.stem.split("-")[-1])) / 20).float()
        val_bc = torch.tensor((int(B_bc.stem.split("-")[-1]) - int(A_bc.stem.split("-")[-1])) / 20).float()
        val_vb = torch.tensor((int(B_vb.stem.split("-")[-1]) - int(A_vb.stem.split("-")[-1])) / 20).float()

        image_ex = ImageAnnotation(path=A_ex, root=self.root)
        ref_ex   = ImageAnnotation(path=B_ex, root=self.root)
        image_bc = ImageAnnotation(path=A_bc, root=self.root)
        ref_bc   = ImageAnnotation(path=B_bc, root=self.root)
        image_vb = ImageAnnotation(path=A_vb, root=self.root)
        ref_vb   = ImageAnnotation(path=B_vb, root=self.root)

        datapoint = {
            "image_ex": image_ex.data,
            "image_bc": image_bc.data,
            "image_vb": image_vb.data,
            "ref_ex"  : ref_ex.data,
            "ref_bc"  : ref_bc.data,
            "ref_vb"  : ref_vb.data,
        }
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = getattr(self.datapoint_attrs[k], "to_tensor", None)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, keepdim=False, normalize=True)
        datapoint |= {
            "val_ex"       : val_ex,
            "val_bc"       : val_bc,
            "val_vb"       : val_vb,
            "image_ex_meta": image_ex.meta,
            "image_bc_meta": image_bc.meta,
            "image_vb_meta": image_vb.meta,
            "ref_ex_meta"  : ref_ex.meta,
            "ref_bc_meta"  : ref_bc.meta,
            "ref_vb_meta"  : ref_vb.meta,
        }

        return datapoint

    def __len__(self) -> int:
        """Returns number of datapoints."""
        return len(self.file_keys)

    def list_data(self):
        """Lists file lists with image annotations for split."""
        patterns = [self.root / "retouch_init"]

        file_ex = defaultdict(list)
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("exposure/*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} exposure images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        idx_ex = path.stem.split("-")[0]
                        file_ex[idx_ex].append(path)

        file_bc = defaultdict(list)
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("black_clipping/*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} black clipping images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        idx_bc = path.stem.split("-")[0]
                        file_bc[idx_bc].append(path)

        file_vb = defaultdict(list)
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("vibrance/*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} vibrance images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        idx_vb = path.stem.split("-")[0]
                        file_vb[idx_vb].append(path)

        self.file_ex   = file_ex
        self.file_bc   = file_bc
        self.file_vb   = file_vb
        self.file_keys = list(self.file_ex.keys())

    def verify_data(self):
        """Verifies dataset (placeholder, no action taken)."""
        pass


@DATASETS.register(name="fivek")
class FiveK(VisionDataset):
    """Loads FiveK dataset from ``root`` dir with Expert A GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

@DATASETS.register(name="fiveka")
class FiveKA(VisionDataset):
    """Loads FiveKA dataset from ``root`` dir with Expert A GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    def list_reference_image(self):
        """Lists ``datapoints`` with reference image annotations."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                for img in pbar.track(sequence=images, description=desc):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref_a/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images
    

@DATASETS.register(name="fivekb")
class FiveKB(VisionDataset):
    """Loads FiveKB dataset from ``root`` dir with Expert B GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    def list_reference_image(self):
        """Lists ``datapoints`` with reference image annotations."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                for img in pbar.track(sequence=images, description=desc):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref_b/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images
            

@DATASETS.register(name="fivekc")
class FiveKC(VisionDataset):
    """Loads FiveKC dataset from ``root`` dir with Expert C GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    def list_reference_image(self):
        """Lists ``datapoints`` with reference image annotations."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                for img in pbar.track(sequence=images, description=desc):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref_c/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images
            

@DATASETS.register(name="fivekd")
class FiveKD(VisionDataset):
    """Loads FiveKD dataset from ``root`` dir with Expert D GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    def list_reference_image(self):
        """Lists ``datapoints`` with reference image annotations."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                for img in pbar.track(sequence=images, description=desc):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref_d/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images
            

@DATASETS.register(name="fiveke")
class FiveKE(VisionDataset):
    """Loads FiveKE dataset from ``root`` dir with Expert E GT.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "fivek" if root.name != "fivek" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    def list_reference_image(self):
        """Lists ``datapoints`` with reference image annotations."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                for img in pbar.track(sequence=images, description=desc):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref_e/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images


# ----- DataModule -----
@DATAMODULES.register(name="fivekinit")
class FiveKInitDataModule(core.DataModule):
    """Configures FiveKInit datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKInit(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = None
        if stage in [None, "test"]:
            self.test  = None

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek")
class FiveKDataModule(core.DataModule):
    """Configures FiveK datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveK(split=Split.TEST, **self.dataset_kwargs)
            self.val   = FiveK(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveK(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fiveka")
class FiveKADataModule(core.DataModule):
    """Configures FiveKA datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKA(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKA(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKA(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivekb")
class FiveKBDataModule(core.DataModule):
    """Configures FiveKB datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKB(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKB(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKB(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivekc")
class FiveKCDataModule(core.DataModule):
    """Configures FiveKC datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKC(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKC(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKC(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivekd")
class FiveKDDataModule(core.DataModule):
    """Configures FiveKD datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKD(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKD(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKD(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fiveke")
class FiveKEDataModule(core.DataModule):
    """Configures FiveKE datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = FiveKE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKE(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKE(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
