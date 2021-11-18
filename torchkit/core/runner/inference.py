#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference Pipeline
"""

from __future__ import annotations

import logging
import os
from typing import Any
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
import torchvision
from pytorch_lightning.utilities import rank_zero_warn
from tqdm import tqdm

from torchkit.core.fileio import create_dirs
from torchkit.core.image import FrameLoader
from torchkit.core.image import ImageWriter
from torchkit.core.image import reshape_image
from torchkit.core.image import resize_image
from torchkit.core.image import unnormalize_image
from torchkit.core.utils import Arrays
from torchkit.core.utils import select_device
from torchkit.core.utils import Tensors
from torchkit.core.utils import to_4d_array
from .utils import get_next_version

logger = logging.getLogger()


# MARK: - Inference

# noinspection PyMethodMayBeStatic
class Inference:
    """The Inference class defines the prediction loop for image data: images,
    folders, video, etc.
    
    Attributes:
        default_root_dir (str):
            The root dir to save predicted data.
        output_dir (str):
            The output directory to save predicted images.
        model (nn.Module):
            The model to run.
        data (str):
            The data source. Can be a path or pattern to image/video/directory.
        data_loader (Any):
            The data loader object.
        shape (tuple, optional):
            The input and output shape of the image as [H, W, C]. If `None`,
            use the input image shape.
        batch_size (int):
            The batch size. Default: `1`.
        device (int, str, optional):
            Will be mapped to either gpus, tpu_cores, num_processes or ipus,
            based on the accelerator type.
        verbose (bool):
            Verbosity mode. Default: `False`.
        save_image (bool):
            Save predicted images. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        default_root_dir: str,
        version         : Union[int, str, None] = None,
        shape           : Optional[tuple]       = None,
        batch_size      : int                   = 1,
        device          : Union[int, str, None] = 0,
        verbose         : bool                  = True,
        save_image      : bool                  = False,
        *args, **kwargs
    ):
        super().__init__()
        self.default_root_dir = default_root_dir
        self.shape            = shape
        self.batch_size       = batch_size
        self.device           = select_device(device=device)
        self.verbose          = verbose
        self.save_image       = save_image
        self.model            = None
        self.post_model       = None
        self.data             = None
        self.data_loader      = None
        self.image_writer     = None
        
        self.init_output_dir(version=version)
        
    # MARK: Configure
    
    def init_output_dir(self, version: Union[int, str, None] = None):
        """Configure output directory base on the given version.
        
        Args:
            version (int, str, optional):
                The experiment version. If version is not specified the logger
                inspects the save directory for existing versions, then
                automatically assigns the next available version. If it is a
                string then it is used as the run-specific subdirectory name,
                otherwise `version_${version}` is used.
        """
        if version is None:
            version = get_next_version(root_dir=self.default_root_dir)
        if isinstance(version, int):
            version = f"version_{version}"
        version = version.lower()
        
        self.output_dir = os.path.join(self.default_root_dir, version)
        rank_zero_warn(f"Output directory at: {self.output_dir}.")
    
    def init_data_loader(self):
        """Configure the data loader object.
        """
        self.data_loader = FrameLoader(
            data=self.data, batch_size=self.batch_size
        )
        
    def init_data_writer(self):
        """Configure the data writer object.
        """
        self.image_writer = ImageWriter(dst=self.output_dir)
        
    def validate_attributes(self):
        """Validate all attributes' values before run loop start.
        """
        assert self.model       is not None, f"Invalid model."
        assert self.data_loader is not None, f"Invalid data loader."
        
        if self.save_image:
            assert self.image_writer is not None, f"Invalid image writer."
        
    # MARK: Run
    
    def run(self, model: Any, data: str, post_model: Any = None):
        """Main prediction loop.
        
        Args:
            model (nn.Module):
                The model to run.
            data (str):
                The data source. Can be a path or pattern to
                image/video/directory.
            post_model (nn.Module):
                The post-processing model.
        """
        self.model      = model
        self.post_model = post_model
        self.data       = data
        
        self.run_routine_start()
        
        # NOTE: Mains loop
        pbar = tqdm(total=len(self.data_loader), desc=f"{self.model.fullname}")
        for batch_idx, batch in enumerate(self.data_loader):
            images, indexes, files, rel_paths = batch
            
            x       = self.preprocess(images)
            y_hat   = self.model.forward(x=x)
            results = self.model.prepare_results(x=x, y_hat=y_hat)
            if self.post_model:
                # results = results[0]
                results = self.post_model.forward(x=results)
                results = self.model.prepare_results(x=x, y_hat=results)
                
            results = self.postprocess(results)
            
            if self.verbose:
                self.show_results(results=results, images=images)
            if self.save_image:
                self.image_writer.write_images(
                    images=results, # image_files=rel_paths
                )
           
            pbar.update(1)
        
        self.run_routine_end()
        
    def run_routine_start(self):
        """When run routine starts we build the `output_dir` on the fly.
        """
        create_dirs(paths=[self.output_dir])
        self.init_data_loader()
        self.init_data_writer()
        self.validate_attributes()

        self.model.to(self.device)
        self.model.eval()
        if self.post_model:
            self.post_model.to(self.device)
            self.post_model.eval()
        
        if self.verbose:
            cv2.namedWindow("results", cv2.WINDOW_KEEPRATIO)

    def run_routine_end(self):
        """When run routine ends we release data loader and writers.
        """
        self.model.train()
        
        if self.verbose:
            cv2.destroyAllWindows()

    def preprocess(self, images: Arrays) -> torch.Tensor:
        """Preprocessing input.

        Args:
            images (Arrays):
                The input images as [B, H, W, C].

        Returns:
        	x (torch.Tensor):
        	    The input tensor as  [B, C H, W].
        """
        x = images
        if self.shape:
            x = [resize_image(image, self.shape)[0] for image in x]
        x = [torchvision.transforms.ToTensor()(image) for image in x]
        x = torch.stack(x)
        x = x.to(self.device)
        return x

    def postprocess(self, results: Tensors) -> np.ndarray:
        """Postprocessing results.

        Args:
            results (Tensors):
                The output images.

        Returns:
            results (np.ndarray):
                The postprocessed output images as [B, H, W, C].
        """
        results = to_4d_array(results)  # List of 4D-array
        results = reshape_image(results, False)
        results = unnormalize_image(results)
        return results

    # MARK: Visualize

    def show_results(
        self, results: np.ndarray, images: Optional[np.ndarray] = None
    ):
        """Show results.
        
        Args:
            results (np.ndarray):
                The postprocessed output images as [B, H, W, C].
            images (np.ndarray, optional):
                The original images. Default: None.
        """
        for img in images:
            cv2.imshow("image", img)
            cv2.waitKey(1)
        for img in results:
            cv2.imshow("results", img)
            cv2.waitKey(1)
