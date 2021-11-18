#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for class-label. A class-label is a dictionary of  label's
properties defined in the dataset. It is not used to defined the labels
detected from the model.
"""

from __future__ import annotations

import logging
from typing import Optional
from typing import Union

import cv2
import numpy as np
from munch import Munch

from torchkit.core.fileio import load

logger = logging.getLogger()


# MARK: - Find

def majority_voting(object_labels: list[dict]) -> dict:
    """Get label that has max appearances in the object's labels list."""
    # NOTE: Count number of appearance of each label.
    unique_labels = Munch()
    label_voting  = Munch()
    for label in object_labels:
        key   = label.get("id")
        value = label_voting.get(key)
        if value:
            label_voting[key] = value + 1
        else:
            unique_labels[key] = label
            label_voting[key]  = 1
    
    # NOTE: get key (label's id) with max value
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]


# MARK: - ClassLabels

class ClassLabels:
    """ClassLabels object is a wrapper around a list of label dictionaries.
    It takes care of all the hassle when working with labels.

    Attributes:
        classlabels (list):
            The list of all classlabels.
    """

    # MARK: Magic Functions

    def __init__(self, classlabels: list):
        self._classlabels = classlabels

    # MARK: Configure

    @staticmethod
    def create_from_dict(label_dict: dict) -> ClassLabels:
        """Create a `ClassLabels` object from a dictionary that contains all
        classlabels.
        """
        if hasattr(label_dict, "classlabels"):
            classlabels = label_dict.get("classlabels")
            classlabels = Munch.fromDict(classlabels)
            return ClassLabels(classlabels=classlabels)
        else:
            raise ValueError(f"Cannot defined labels!")
    
    @staticmethod
    def create_from_file(label_path: str) -> ClassLabels:
        """Create a `ClassLabels` object from a file that contains all
        classlabels
        """
        labels_dict = load(path=label_path)
        classlabels = labels_dict["classlabels"]
        classlabels = Munch.fromDict(classlabels)
        return ClassLabels(classlabels=classlabels)
        
    # MARK: Property

    @property
    def classlabels(self) -> list:
        """Return the list of all labels' dictionaries."""
        return self._classlabels

    def color_legend(self, height: Optional[int] = None) -> np.ndarray:
        """Return a color legend using OpenCV drawing functions.

		References:
			https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/

		Args:
			height (int, optional):
				Height of the color legend image. Defaults: `None`.

		Returns:
			legend (np.ndarray):
				The color legend image.
		"""
        num_classes = len(self.classlabels)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(
            ((num_classes * row_height) + 25, 300, 3), dtype=np.uint8
        )

        # NOTE: Loop over the class names + colors
        for i, label in enumerate(self.classlabels):
            # Draw the class name + color on the legend
            color = label.color
            # Convert to BGR format since OpenCV operates on BGR format.
            color = color[::-1]
            cv2.putText(
                img       = legend,
                text      = label.name,
                org       = (5, (i * row_height) + 17),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color     = (0, 0, 255),
                thickness = 2
            )
            cv2.rectangle(
                img       = legend,
                pt1       = (150, (i * 25)),
                pt2       = (300, (i * row_height) + 25),
                color     = color,
                thickness = -1
            )
        return legend
        
    def colors(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all labels' colors.
        
        Args:
            key (str):
                The label's key to search from `labels`
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
            exclude_max_key (bool):
			    If `True` only count class's label with key < 255.
        """
        labels_colors = []
        for label in self.classlabels:
            if hasattr(label, key) and hasattr(label, "color"):
                if (exclude_negative_key and label[key] < 0   ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_colors.append(label.color)

        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """Return a dictionary of id to label object."""
        return {label["id"]: label for label in self.classlabels}

    def ids(
        self,
        key                 : str = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all labels' ids at `key`.
        
        Args:
            key (str):
                The label's key to search from `labels`.
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
            exclude_max_key (bool):
                If `True` only count class's label with key < 255.
        """
        labels_ids = []
        for label in self.classlabels:
            if hasattr(label, key):
                if (exclude_negative_key and label[key] < 0   ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_ids.append(label[key])

        return labels_ids

    @property
    def list(self) -> list:
        """Alias to `classlabels()`."""
        return self.classlabels

    @property
    def name2label(self) -> dict[str, dict]:
        """Return a dictionary of {`name`: `label object`}."""
        return {label["name"]: label for label in self.classlabels}
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> int:
        """Return the number of classes.

        Args:
            key (str):
                The label's key to search from `labels`. Defaults: `id`.
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0. Defaults: `True`.
            exclude_max_key (bool):
			    If `True` only count class's label with key < 255. Defaults: `True`.
        """
        count = 0
        for classlabels in self.classlabels:
            if hasattr(classlabels, key):
                if (exclude_negative_key and classlabels[key] < 0   ) or \
                   (exclude_max_key      and classlabels[key] >= 255):
                    continue
                count += 1
        return count

    # MARK: Custom Accessors

    def get_classlabel(
        self,
        key  : str                   = "id",
        value: Union[int, str, None] = None
    ) -> Optional[dict]:
        """Get the classlabel based on the given (`key`, `value`) pair."""
        for classlabel in self.classlabels:
            if hasattr(classlabel, key) and (value == classlabel[key]):
                return classlabel
        return None
    
    def get_classlabel_by_name(self, name: str) -> Optional[dict]:
        """Get the classlabel based on the given `name`."""
        return self.get_classlabel(key="name", value=name)
    
    def get_id(self, name: str) -> Optional[int]:
        """Get the id based on the given `name`."""
        classlabel: dict = self.get_classlabel_by_name(name=name)
        return classlabel["id"] if classlabel is not None else None
    
    def get_name(
        self,
        key  : str                   = "id",
        value: Union[int, str, None] = None
    ) -> Optional[str]:
        """Get the classlabel's name based on the given (`key`, `value`)
        pair.
        """
        classlabel: dict = self.get_classlabel(key=key, value=value)
        return classlabel["name"] if classlabel is not None else None
    
    # MARK: Visualize

    def show_color_legend(self, height: Optional[int] = None):
        """Show a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height (int, optional):
        		Height of the color legend image.
        """
        color_legend = self.color_legend(height=height)
        cv2.imshow(winname="Color Legend", mat=color_legend)
        cv2.waitKey(1)
