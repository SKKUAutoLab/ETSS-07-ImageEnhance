#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Debugger to save results during training.
"""

from __future__ import annotations

import logging
import threading
from queue import Queue
from typing import Optional
from typing import Union

from torchkit.core.utils import Arrays
from torchkit.core.utils import FuncCls
from torchkit.core.utils import Tensors

logger = logging.getLogger()


# MARK: - Debugger

class Debugger:
    """
    
    Attributes:
        every_n_epochs (int):
            Number of epochs between debugging. To disable, set
            `every_n_epochs=0`. Default: `1`.
        run_in_parallel (bool):
            If `True` runs debugging process in a separated thread.
            Default: `True`.
        queue_size (int):
            The debug queue size. It should equal the value of `save_max_n`.
            Default: `20`.
        save_max_n (int):
            Maximum debugging items to be kept. Default: `20`.
        save_to_subdir (bool):
            Save all debug images of the same epoch to a sub-directory naming
		    after the epoch number. Default: `True`.
        image_quality (int):
            The image quality to be saved. Default: `95`.
        show (bool):
            If `True` shows the results on the screen. Default: `False`.
        show_max_n (int):
            Maximum debugging items to be shown. Default: `8`.
        show_func (FunCls, optional):
            Function to visualize the debug results. Default: `None`.
        wait_time (float):
            Pause some times before showing the next image. Default: `0.001`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        every_n_epochs : int               = 1,
        run_in_parallel: bool              = True,
        queue_size     : Optional[int]     = 20,
        save_max_n     : int               = 20,
        save_to_subdir : bool              = True,
        image_quality  : int               = 95,
        show           : bool              = False,
        show_max_n     : int               = 8,
        show_func      : Optional[FuncCls] = None,
        wait_time      : float             = 0.001,
        *args, **kwargs
    ):
        super().__init__()
        self.every_n_epochs  = every_n_epochs
        self.run_in_parallel = run_in_parallel
        self.queue_size      = queue_size
        self.save_max_n      = save_max_n
        self.save_to_subdir  = save_to_subdir
        self.image_quality   = image_quality
        self.show            = show
        self.show_max_n      = show_max_n
        self.show_func       = show_func
        self.wait_time       = wait_time
        self.debug_queue     = None
        self.thread_debugger = None
        
        self.init_thread()
        
    # MARK: Configure
    
    def init_thread(self):
        if self.run_in_parallel:
            self.debug_queue     = Queue(maxsize=self.queue_size)
            self.thread_debugger = threading.Thread(
                target=self.show_results_parallel
            )
            
    # MARK: Run
    
    def run(
        self,
        x       : Union[Tensors, Arrays],
        y    	: Optional[Union[Tensors, Arrays]] = None,
        y_hat	: Optional[Union[Tensors, Arrays]] = None,
        filepath: Optional[str] 				   = None,
    ):
        """Run the debugger process."""
        if self.show_func:
            if self.thread_debugger:
                self.debug_queue.put([x, y, y_hat, filepath])
            else:
                self.show_results(x=x, y=y, y_hat=y_hat, filepath=filepath)

    def run_routine_start(self):
        """Perform operations when run routine starts."""
        if self.thread_debugger:
            self.thread_debugger.start()
    
    def run_routine_end(self):
        """Perform operations when run routine ends."""
        if self.thread_debugger:
            self.debug_queue.put([None, None, None, None])
    
    # MARK: Visualize

    def show_results(
        self,
        x       : Union[Tensors, Arrays],
        y    	: Optional[Union[Tensors, Arrays]] = None,
        y_hat	: Optional[Union[Tensors, Arrays]] = None,
        filepath: Optional[str] 				   = None,
        *args, **kwargs
    ):
        self.show_func(
            x=x, y=y, y_hat=y_hat, filepath=filepath,
            image_quality=self.image_quality, show=self.show,
            show_max_n=self.show_max_n, wait_time=self.wait_time,
            *args, **kwargs
        )

    def show_results_parallel(self):
        """Draw `result` in a separated thread."""
        while True:
            (x, y, y_hat, filepath) = self.debug_queue.get()
            if x is None:
                break
            self.show_results(x=x, y=y, y_hat=y_hat, filepath=filepath)
    
        # Stop debugger thread
        self.thread_debugger.join()
