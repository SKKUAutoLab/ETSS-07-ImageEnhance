#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for all datasets."""

__all__ = [
    "DatapointAttributes",
    "VisionDataset",
]

from abc import ABC
from typing import Any, Literal

from mon import core
from mon.constants import DepthDataSource, InfraredDataSource, TRANSFORMS
from mon.vision.geometry import albumentation
from mon.vision.types.depth import DepthMapAnnotation
from mon.vision.types.image import ImageAnnotation
from mon.vision.types.thermal import InfraredAnnotation

DatapointAttributes = core.DatapointAttributes


# ----- Vision Dataset -----
class VisionDataset(core.Dataset, ABC):
    """Base class for multimodal, multi-task, multi-label datasets.

    Attributes:
        datapoint_attrs: Dict of attribute names and types. Common attributes:
            - ``'image'``    : ``ImageAnnotation`` (main attribute)
            - ``'depth'``    : ``DepthMapAnnotation``
            - ``'ref_image'``: ``ImageAnnotation``
            - ``'ref_depth'``: ``DepthMapAnnotation``

    Args:
        depth_source: Source of depth data. Default is ``'dav2_vitb'``.
        infrared_source: Source of infrared data. Default is ``'infrared'``.
    """
    
    def __init__(
        self,
        depth_source   : Literal[*DepthDataSource.values()]    = "dav2_vitb",
        infrared_source: Literal[*InfraredDataSource.values()] = "infrared",
        *args, **kwargs
    ):
        if depth_source not in DepthDataSource:
            raise ValueError(f"[depth_source] must be one of {DepthDataSource}, got {depth_source}.")
        self.depth_source = depth_source
        
        if infrared_source not in InfraredDataSource:
            raise ValueError(f"[infrared_source] must be one of {InfraredDataSource}, got {infrared_source}.")
        self.infrared_source = infrared_source
        super().__init__(*args, **kwargs)
    
    # ----- Magic Methods -----
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at given ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            ``dict`` with datapoint and metadata.
        """
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            main_attr      = self.main_attribute
            args           = {k: v for k, v in datapoint.items() if v is not None}
            args["image"]  = args.pop(main_attr)
            transformed    = self.transform(**args)
            transformed[main_attr] = transformed.pop("image")
            datapoint     |= transformed
        
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = getattr(self.datapoint_attrs[k], "to_tensor", None)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, normalize=True)
        
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Gets total number of datapoints.

        Returns:
            Number of datapoints in dataset.
        """
        return len(self.datapoints[self.main_attribute])
    
    # ----- Properties -----
    @property
    def albumentation_target_types(self) -> dict[str, str]:
        """Gets the target types for Albumentations.
        
        Returns:
            ``dict`` with keys as attribute names and values as target types.
        """
        target_types = {}
        for k, v in self.datapoint_attrs.items():
            target_type = getattr(v, "albumentation_target_type", None)
            if target_type:
                target_types[k] = target_type
        
        target_types.pop("meta", None)
        return target_types
    
    # ----- Initialize -----
    def init_transform(self, transform: albumentation.Compose | Any = None):
        """Initializes transformations with multimodal support.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        if transform is None:
            self.transform = None
        elif isinstance(transform, albumentation.Compose):
            self.transform = transform
        else:
            transform  = [transform] if not isinstance(transform, (list, tuple)) else transform
            transform_ = []
            for t in transform:
                if isinstance(t, albumentation.BasicTransform):
                    transform_.append(t)
                elif isinstance(t, dict):
                    t_ = TRANSFORMS.build(config=t)
                    if t_:
                        transform_.append(t_)
                    else:
                        raise ValueError(f"Transform [{t}] is not supported.")
                else:
                    raise TypeError(f"[transform] must be a list of albumentation.BasicTransform "
                                    f"or dicts, got {type(t)}.")
            self.transform = albumentation.Compose(transforms=transform_)
            
        if isinstance(self.transform, albumentation.Compose):
            additional_targets = self.albumentation_target_types
            additional_targets.pop(self.main_attribute, None)
            self.transform.add_targets(additional_targets)
    
    def init_data(self, cache_data: bool = False):
        """Initializes dataset data with multimodal support.

        Args:
            cache_data: If ``True``, caches data to disk. Default is ``False``.
        """
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.list_data()
            self.list_multimodal_data()
        
        self.filter_data()
        self.verify_data()
        
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            core.delete_cache(cache_file)
    
    def list_multimodal_data(self):
        """Lists multimodal data for the dataset."""
        if "depth" in self.datapoint_attrs:
            self.list_depth_map()
        if "infrared" in self.datapoint_attrs:
            self.list_infrared_map()
        
        if self.has_annotations:
            self.list_reference_image()
            if "ref_depth" in self.datapoint_attrs:
                self.list_reference_depth_map()
            if "ref_infrared" in self.datapoint_attrs:
                self.list_reference_infrared_map()
        else:
            self.datapoints.pop("ref_image",    None)
            self.datapoints.pop("ref_depth",    None)
            self.datapoints.pop("ref_infrared", None)

    def list_depth_map(self):
        """Lists depth maps for the dataset."""
        images = self.datapoints.get("image", [])
        depths = self.datapoints.get("depth", [])
        
        if len(images) > 0 and len(depths) == 0:
            depths: list[DepthMapAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.depth_source}/")
                    depths.append(
                        DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["depth"] = depths

    def list_infrared_map(self):
        """Lists infrared maps for the dataset."""
        images    = self.datapoints.get("image",    [])
        infrareds = self.datapoints.get("infrared", [])

        if len(images) > 0 and len(infrareds) == 0:
            infrareds: list[InfraredAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} infrared maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.infrared_source}/")
                    infrareds.append(
                        InfraredAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.infrared_source
                        )
                    )
            self.datapoints["infrared"] = infrareds

    def list_bboxes(self):
        """Lists bboxes for the dataset."""
        images = self.datapoints.get("image",  [])
        bboxes = self.datapoints.get("bboxes", [])

        if len(images) > 0 and len(bboxes) == 0:
            bboxes: list[InfraredAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} bboxes"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/label/")
                    bboxes.append(
                        InfraredAnnotation(
                            path   = path.txt_file(),
                            root   = img.root,
                            source = self.infrared_source
                        )
                    )
            self.datapoints["bboxes"] = bboxes

    def list_reference_image(self):
        """Lists reference images for the dataset."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])

        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference images"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref/")
                    ref_images.append(ImageAnnotation(
                        path = path.image_file(),
                        root = img.root
                    ))
            self.datapoints["ref_image"] = ref_images

    def list_reference_depth_map(self):
        """Lists reference depth maps for the dataset."""
        ref_images = self.datapoints.get("ref_image", [])
        ref_depths = self.datapoints.get("ref_depth", [])
        
        if len(ref_images) > 0 and len(ref_depths) == 0:
            ref_depths: list[DepthMapAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = ref_images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.depth_source}/")
                    ref_depths.append(
                        DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["ref_depth"] = ref_depths

    def list_reference_infrared_map(self):
        """Lists reference infrared maps for the dataset."""
        ref_images    = self.datapoints.get("ref_image",    [])
        ref_infrareds = self.datapoints.get("ref_infrared", [])
        
        if len(ref_images) > 0 and len(ref_infrareds) == 0:
            ref_infrareds: list[InfraredAnnotation] = []
            with core.create_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = ref_images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference infrared maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.infrared_source}/")
                    ref_infrareds.append(
                        InfraredAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["ref_infrared"] = ref_infrareds
    
    def filter_data(self):
        """Filter unwanted datapoints."""
        pass
    
    def verify_data(self):
        """Verifies dataset integrity.

        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        for k, v in self.datapoints.items():
            if k not in self.datapoint_attrs:
                raise RuntimeError(f"Attribute [{k}] is not defined in [datapoint_attrs]; "
                                   f"define it in the class if intentional.")
            if self.datapoint_attrs[k]:
                if v is None:
                    raise RuntimeError(f"No [{k}] attributes defined")
                if v is not None and len(v) != self.__len__():
                    raise RuntimeError(f"Number of [{k}] attributes ({len(v)}) does not "
                                       f"match datapoints ({self.__len__()}).")
        if self.verbose:
            core.console.log(f"Number of {self.split_str} datapoints: {self.__len__()}")
    
    def reset(self):
        """Resets the dataset to start over."""
        self.index = 0
    
    def close(self):
        """Closes and releases dataset resources."""
        pass
    
    # ----- Data Retrieval -----
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict containing datapoint data.
        """
        datapoint = self.new_datapoint
        for k, v in self.datapoints.items():
            if v is not None and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
        return datapoint
    
    def get_meta(self, index: int) -> dict:
        """Gets metadata at specified index.

        Args:
            index: Index of metadata.

        Returns:
            Dict with metadata from main attribute.
        """
        return self.datapoints[self.main_attribute][index].meta
