#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import os

from exps.utils import checkpoints_dir

__all__ = ["config"]

# MARK: - Notes

"""
NOTES:
	- MBLLEN model requires input shape to be [:, 256, 256].
	- Optimizer should be: dict(name="adam", lr=0.0002)
"""


# MARK: - Basic Info

model_name = "mprnet"
# The model's name. Also, the root folder located inside `models_zoo_dir`.
data_name = "mprnet_snow"
# The trained data name.
model_fullname = f"{model_name}_{data_name}"
# It represents the model with trained dataset.
version = 0
# Experiment version.


# MARK: - Dirs

root_dir  = model_name
model_dir = os.path.join(checkpoints_dir, root_dir, model_fullname)


# MARK: - Configs

checkpoint = {
    "model_dir": model_dir,
    # The model's dir. The checkpoints will be save to
	# `../<model_dir>/<version>/weights/`.
    "version": version,
    # Experiment version. If version is not specified the logger inspects the
	# save directory for existing versions, then automatically assigns the
	# next available version. If it is a string then it is used as the
	# run-specific subdirectory name, otherwise `version_${version}` is used.
    "filename": None,
    # Checkpoint filename. Can contain named formatting options to be
	# auto-filled. If `None`, it will be set to `epoch={epoch}.ckpt`.
	# Default: `None`.
    "auto_insert_metric_name": True,
    # When `True`, the checkpoints filenames will contain the metric name.
	# Default: `True`.
    "monitor": "psnr_epoch",  # "loss_epoch",
    # Quantity to monitor. Default: `None` which will monitor `loss_epoch`.
    "mode": "max",
    # One of: [`min`, `max`]. For `acc`, this should be `max`, for `loss`
	# this should be `min`, etc.
    "verbose": True,
    # Verbosity mode. Default: `False`.
    "save_weights_only": False,
    # If `True`, then only the model’s weights will be saved
	# `model.save_weights(filepath)`, else the full model is saved
	# `model.save(filepath)`.
    "every_n_train_steps": None,
    # Number of training steps between checkpoints.
    # If `every_n_train_steps == None or every_n_train_steps == 0`, we skip
	# saving during training. To disable, set `every_n_train_steps = 0`. This
	# value must be `None` or non-negative. This must be mutually exclusive
	# with `train_time_interval` and `every_n_epochs`. Default: `None`.
    "every_n_epochs": 1,
    # Number of epochs between checkpoints. If `every_n_epochs == None` or
	# `every_n_epochs == 0`, we skip saving when the epoch ends. To disable,
	# `set every_n_epochs = 0`. This value must be None or non-negative.
	# Default: `1`.
    "train_time_interval": None,
    # Checkpoints are monitored at the specified time interval. For all
	# practical purposes, this cannot be smaller than the amount of time it
	# takes to process a single training batch. This is not guaranteed to
	# execute at the exact time specified, but should be close. This must be
	# mutually exclusive with `every_n_train_steps` and `every_n_epochs`.
	# Default: `None`.
    "save_on_train_epoch_end": True
    # Whether to run checkpointing at the end of the training epoch. If this
	# is `False`, then the check runs at the end of the validation. If `None`
	# then skip saving. Default: `False`.
}

tb_logger = {
    "save_dir": model_dir,
    # Save directory.
    "name": "",
    # Experiment name. Default: `default`. If it is the empty string then no
	# per-experiment subdirectory is used.
    "version": version,
    # Experiment version. If version is not specified the logger inspects the
	# save directory for existing versions, then automatically assigns the
	# next available version. If it is a string then it is used as the
	# run-specific subdirectory name, otherwise `version_${version}` is used.
    "sub_dir": None,
    # Sub-directory to group TensorBoard logs. If a sub_dir argument is
	# passed then logs are saved in `/save_dir/version/sub_dir/`. Default:
	# `None` in which logs are saved in `/save_dir/version/`.
    "log_graph": False,
    # Adds the computational graph to tensorboard. This requires that the
	# user has defined the
    # `self.example_input_array` attribute in their model.
    "default_hp_metric": True,
    # Enables a placeholder metric with key `hp_metric` when
	# `log_hyperparams` is called without a metric (otherwise calls to
	# log_hyperparams without a metric are ignored).
    "prefix": "",
    # A string to put at the beginning of metric keys.
}

trainer = {
	"accumulate_grad_batches": None,
	# Accumulates grads every k batches or as set up in the dict.
	# Default: `None`.
	"amp_backend": "apex",
	# The mixed precision backend to use (`native` or `apex`).
	# Default: `native`.
	"amp_level": None,
	# The optimization level to use (O1, O2, etc...). By default it will be set
	# to "O2" if `amp_backend` is set to `apex`.
	"auto_lr_find": False,
	# If set to `True`, will make trainer.tune() run a learning rate finder,
	# trying to optimize initial learning for faster convergence.
	# trainer.tune() method will set the suggested learning rate in self.lr
	# or self.learning_rate in the LightningModule. To use a different key
	# set a string instead of True with the key name. Default: `False`.
	"auto_scale_batch_size": False,
	# If set to `True`, will initially run a batch size finder trying to find
	# the largest batch size that fits into memory. The result will be stored
	# in self.batch_size in the LightningModule. Additionally, can be set to
	# either power that estimates the batch size through a power search or
	# binsearch that estimates the batch size through a binary search.
	# Default: `False`.
	"auto_select_gpus": False,
	# If enabled and `gpus` is an integer, pick available gpus automatically.
	# This is especially useful when GPUs are configured to be in “exclusive
	# mode”, such that only one process at a time can access them.
	# Default: `False`
	"benchmark": False,
	# If `True` enables cudnn.benchmark. Default: `False`.
	"callbacks": None,
	# Add a callback or list of callbacks. Default: `None`, will be defined
	# when in code.
	"check_val_every_n_epoch": 1,
	# Check val every n train epochs. Default: `1`.
	"default_root_dir": None,
	# Default path for logs and weights when no logger/ckpt_callback passed.
	# Default: `None`.
	"detect_anomaly": False,
	# Enable anomaly detection for the autograd engine. Default: `False`.
	"deterministic": False,
	# If true enables cudnn.deterministic. Default: `False`.
	"devices": None,
	# Will be mapped to either gpus, tpu_cores, num_processes or ipus,
	# based on the accelerator type. Default: `None`.
	"enable_checkpointing": False,
	# If `True`, enable checkpointing. It will configure a default
	# ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
	# `callbacks`.
	"enable_model_summary": True,
	# Whether to enable model summarization by default.
	"enable_progress_bar": True,
	# Whether to enable to progress bar by default.
	"fast_dev_run": False,
	# Runs n if set to n (int) else 1 if set to True batch(es) of train,
	# val and test to find any bugs (ie: a sort of unit test). Default: `False`.
	"gpus": None,
	# Number of gpus to train on (int) or which GPUs to train on (list or
	# str) applied per node. Defined at runtime. Default: `None`.
	"gradient_clip_val": None,
	# The value at which to clip gradients. Passing `gradient_clip_val=None`
	# disables gradient clipping. If using Automatic Mixed Precision (AMP), the
	# gradients will be unscaled before. Default: `None`.
	"gradient_clip_algorithm": None,
	# The gradient clipping algorithm to use. Pass
	# `gradient_clip_algorithm="value"` to clip by value,
	# and `gradient_clip_algorithm="norm"` to clip by norm. By default it will
	# be set to `norm`. Default: `None`.
	"ipus": None,
	# How many IPUs to train on. Default: `None`.
	"limit_train_batches": 1.0,
	# How much of training dataset to check
	# (float = fraction, int = num_batches). Default: 1.0.
	"limit_val_batches": 1.0,
	# How much of validation dataset to check
	# (float = fraction, int = num_batches). Default: 1.0.
	"limit_test_batches": 1.0,
	# How much of test dataset to check
	# (float = fraction, int = num_batches). Default: 1.0.
	"limit_predict_batches": 1.0,
	# How much of prediction dataset to check
	# (float = fraction, int = num_batches). Default: 1.0.
	"logger": True,
	# Logger (or iterable collection of loggers) for experiment tracking. A
	# True value uses the default TensorBoardLogger. False will disable
	# logging. If multiple loggers are provided and the save_dir property of
	# that logger is not set, local files (checkpoints, profiler traces,
	# etc.) are saved in default_root_dir rather than in the log_dir of any
	# of the individual loggers. Default: `True`.
	"log_every_n_steps": 50,
	# How often to log within steps. Default: `50`.
	"max_epochs": 200,
	# Stop training once this number of epochs is reached. Disabled by
	# default (None). If both max_epochs and max_steps are not specified,
	# defaults to max_epochs = 1000.
	"max_steps": -1,
	# Stop training after this number of steps. Default: `-1`, disabled.
	"max_time": None,
	# Stop training after this amount of time has passed. Disabled by default
	# (None). The time duration can be specified in the format DD:HH:MM:SS (
	# days, hours, minutes seconds), as a datetime.timedelta, or a dictionary
	# with keys that will be passed to datetime.timedelta. Default: `None`.
	"min_epochs": 1,
	# Force training for at least these many epochs. Disabled by default (
	# None). If both min_epochs and min_steps are not specified, defaults to
	# min_epochs = 1.
	"min_steps": None,
	# Force training for at least these number of steps. Default: `None`,
	# disabled.
	"move_metrics_to_cpu": False,
	# Whether to force internal logged metrics to be moved to cpu. This can
	# save some gpu memory, but can make training slower. Use with attention.
	# Default: `False`.
	"multiple_trainloader_mode": "max_size_cycle",
	# How to loop over the datasets when there are multiple train loaders. In
	# ‘max_size_cycle’ mode, the trainer ends one epoch when the largest
	# dataset is traversed, and smaller datasets reload when running out of
	# their data. In ‘min_size’ mode, all the datasets reload when reaching
	# the minimum length of datasets. Default: `max_size_cycle`.
	"num_nodes": 1,
	# Number of GPU nodes for distributed training. Default: `1`. Defined at
	# runtime. Default: `1`.
	"num_processes": 1,
	# Number of processes for distributed training with
	# distributed_backend=”ddp_cpu”. Defined at runtime. Default: `1`.
	"num_sanity_val_steps": 2,
	# Sanity check runs n validation batches before starting the training
	# routine. Set it to -1 to run all batches in all validation dataloaders.
	# Default: `2`.
	"overfit_batches": 0.0,
	# Overfit a fraction of training data (float) or a set number of batches
	# (int). Default: `0.0`.
	"profiler": None,
	# To profile individual steps during training and assist in identifying
	# bottlenecks. Default: `None`.
	"plugins": None,
	# Plugins allow modification of core behavior like ddp and amp, and enable
	# custom lightning plugins. Default: `None`.
	"precision": 32,
	# Double precision (64), full precision (32), half precision (16) or
	# bfloat16 precision (bf16). Can be used on CPU, GPU or TPUs. Default: `32`.
	"reload_dataloaders_every_n_epochs": 0,
	# Set to a non-negative integer to reload dataloaders every n epochs.
	# Default: `0`.
	"replace_sampler_ddp": True,
	# Explicitly enables or disables sampler replacement. If not specified
	# this will be toggled automatically when DDP is used. By default,
	# it will add shuffle=True for train sampler and shuffle=False for
	# val/test sampler. If you want to customize it, you can set
	# replace_sampler_ddp=False and add your own distributed sampler.
	"strategy": "dp",
	# Previously known as distributed_backend (dp, ddp, ddp2, etc…). Can also
	# take in an accelerator object for custom hardware. Default: `None`.
	# Defined at runtime.
	"sync_batchnorm": False,
	# Synchronize batch norm layers between process groups/whole world.
	# Default: `False`.
	"tpu_cores": None,
	# How many TPU cores to train on (1 or 8) / Single TPU to train on [1].
	# Default: `None`.
	"track_grad_norm": -1,
	# `-1` no tracking. Otherwise, tracks that p-norm. May be set to `inf`
	# infinity-norm. Default: `-1`.
	"val_check_interval": 1.0,
	# How often to check the validation set. Use float to check within a
	# training epoch, use int to check every n steps (batches). Default: `1.0`.
}

inference = {
    # "default_root_dir": infer_dir,
    # The root dir to save predicted data.
    "version": None,
    # The experiment version. If version is not specified the logger inspects
	# the save directory for existing versions, then automatically assigns
	# the next available version. If it is a string then it is used as the
	# run-specific subdirectory name, otherwise `version_${version}` is used.
    "shape": [256, 256, 3],
    # The input and output shape of the image as [H, W, C]. If `None`,
	# use the input image shape.
    "batch_size": 1,
    # The batch size. Default: `1`.
    "verbose": True,
    # Verbosity mode. Default: `False`.
    "save_image": True,
    # Save predicted images. Default: `False`.
}

data = {
    "name": data_name,
    # The datasets" name.
    "subset": ["snow100k"],
    # The type of sub-dataset to use. Can also be a list to include multiple
	# subsets. One of: [`snow100k`, `srrs`, `all`, `*`, `None`]. When `all`,
	# `*` or `None`, all subsets will be included. Default: `*`.
    "snow_size": ["*"],
    # The snow size subset to use. Can also be a list to include multiple
	# subsets. One of: [`s`, `m`, `l`, `*`, `None`]. When `all`, `*` or
	# `None`, all subsets will be included. Default: `*`.
    "shape": [256, 256, 3],
    # The image shape as [H, W, C]. This is compatible with OpenCV format.
    "batch_size": 4,
    # Number of samples in one forward & backward pass.
    "label_format": "custom",
    # The format to convert images and labels to when `get_item()`. Each
	# labels' format has each own directory:
    # annotations_<format>. Example: `.../train/annotations_yolo/...`
    # Supports:
    # - `custom`: uses our custom annotation format.
    # - `coco`  : uses Coco annotation format.
    # - `yolo`  : uses Yolo annotation format.
    # - `pascal`: uses Pascal annotation format.
    # Default: `yolo`.
    "caching_labels": True,
    # Should overwrite the existing cached labels? Default: `False`.
    "caching_images": False,
    # Cache images into memory for faster training. Default: `False`.
    "write_labels": False,
    # After loading images and labels for the first time, we will convert it
	# to our custom data format and write to files. If `True`, we will
	# overwrite these files. Default: `False`.
    "fast_dev_run": False,
    # Take a small subset of the data for fast debug (i.e, like unit
	# testing). Default: `False`.
    "shuffle": True,
    # Set to `True` to have the data reshuffled at every training epoch.
	# Default: `True`.
    "augment": {
        "hsv_h": 0.0,  # Image HSV-Hue augmentation (fraction).
        "hsv_s": 0.0,  # Image HSV-Saturation augmentation (fraction).
        "hsv_v": 0.0,  # Image HSV-Value augmentation (fraction).
        "rotate": 0.0,  # Image rotation (+/- deg).
        "translate": 0.0,  # Image translation (+/- fraction).
        "scale": 0.0,  # Image scale (+/- gain).
        "shear": 0.0,  # Image shear (+/- deg).
        "perspective": 0.0,  # Image perspective (+/- fraction), range 0-0.001.
        "flip_ud": 0.5,  # Image flip up-down (probability).
        "flip_lr": 0.5,  # Image flip left-right (probability).
        "mixup": 0.0,  # Image mixup (probability).
        "mosaic": False,  # Use mosaic augmentation.
        "rect": False,
        # Train model using rectangular images instead of square ones.
        "stride": 32,
        # When `rect_training=True`, reshape the image shapes as a multiply
		# of stride.
        "pad": 0.0,
        # When `rect_training=True`, pad the empty pixel with given values
    }
}

model = {
    "name": model_name,
    # The model's name.
    "fullname": model_fullname,
    # The fullname of the model as: {name}_{data_name}. If `None`,
	# the fullname will be determined when initializing the model.
    "model_dir": model_dir,
    # The model's save dir.
    "version": version,
    # Experiment version. If version is not specified the logger inspects the
	# save directory for existing versions, then automatically assigns the
	# next available version. If it is a string then it is used as the
	# run-specific subdirectory name, otherwise `version_${version}` is used.
    "shape": data["shape"],
    # The image shape as [H, W, C].
    "num_classes": None,
	# Number of classes in the dataset that is used to train the model.
	"classlabels": None,
	# The `ClassLabels` object contains all class-labels defined in the dataset.
	"pretrained": False,
	# Initialize weights from pretrained.
	# - If `True`, use the original pretrained described by the author (
	#   usually, ImageNet or COCO). By default, it is the first element in the
	#   `model_urls` dictionary.
	# - If `str` and is a file/path, then load weights from saved file.
	# - In each inherited model, `pretrained` can be a dictionary's key to
	#   get the corresponding local file or url of the weight.
    "cfg": "B",
	# The config to build the model's layers.
	"out_indexes": -1,
	# The list of layers' indexes to extract features. This is called in
	# `forward_features()` and is useful when the model is used as a
	# component in another model.
	# - If is a `tuple` or `list`, return an array of features.
	# - If is a `int`, return only the feature from that layer's index.
	# - If is `-1`, return the last layer's output.
	# Default: `-1`.
    "loss": dict(name="mpr_loss"),
	# The loss config.
    "metrics": [
		# The list of metrics' configs dictionaries. Default: `None`.
		dict(name="psnr"),
	],
    "optimizers": [
        # The optimizers' configs.
        dict(name="adam", lr=2e-4),
    ],
    "schedulers": [
        # The 2D-list of schedulers' configs. Each sub-list of dicts is
		# corresponded to an optimizer defined in `optimizers`.
        [
            dict(
                name="gradual_warmup_scheduler",
                multiplier=1,
                total_epoch=3,
                after_scheduler=dict(name="cosine_annealing_lr",
                                     T_max=trainer["max_epochs"] - 3,
                                     eta_min=1e-6)
            ),
        ],
    ],
    "debugger": {
		"every_n_epochs": 1,
		# Number of epochs between debugging. To disable, set
		# `every_n_epochs=0`. Default: `1`.
		"run_in_parallel": True,
		# If `True` runs debugging process in a separated thread.
		# Default: `True`.
		"queue_size": 16,
		# The debug queue size.
		"save_max_n": 20,
		# Maximum debugging items to be kept. Default: `50`.
		"save_to_subdir": True,
		# Save all debug images of the same epoch to a sub-directory naming
		# after the epoch number. Default: `True`.
		"image_quality": 95,
		# The image quality to be saved. Default: `95`.
		"show": False,
		# If `True` shows the results on the screen. Default: `False`.
		"show_max_n": 8,
		# Maximum debugging items to be shown. Default: `8`.
		"wait_time": 0.001,
		# Pause some times before showing the next image. Default: `0.001`.
	},
}

config = {
    "checkpoint": checkpoint,
    # The checkpoint config.
    "tb_logger": tb_logger,
    # The tensorboard logger config.
    "trainer": trainer,
    # The trainer config.
    "inference": inference,
    # The inference config.
    "data": data,
    # The dataset config.
    "model": model,
    # The model config.
}


# MARK: - Test

if __name__ == "__main__":
    print(config)
