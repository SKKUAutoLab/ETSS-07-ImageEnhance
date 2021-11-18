#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers import WandbLogger

from .builder import LOGGERS


# MARK: - Builder

LOGGERS.register(name="tensorboard",        module=TensorBoardLogger)
LOGGERS.register(name="tensorboard_logger", module=TensorBoardLogger)
LOGGERS.register(name="TensorBoardLogger",  module=TensorBoardLogger)
LOGGERS.register(name="csv",                module=CSVLogger)
LOGGERS.register(name="csv_logger",         module=CSVLogger)
LOGGERS.register(name="CSVLogger",          module=CSVLogger)
LOGGERS.register(name="comet",              module=CometLogger)
LOGGERS.register(name="comet_logger",       module=CometLogger)
LOGGERS.register(name="CometLogger",        module=CometLogger)
LOGGERS.register(name="wandb",              module=WandbLogger)
LOGGERS.register(name="wandb_logger",       module=WandbLogger)
LOGGERS.register(name="WandbLogger",        module=WandbLogger)
LOGGERS.register(name="mlflow",             module=MLFlowLogger)
LOGGERS.register(name="mlflow_logger",      module=MLFlowLogger)
LOGGERS.register(name="MLFlowLogger",       module=MLFlowLogger)
LOGGERS.register(name="neptune",            module=NeptuneLogger)
LOGGERS.register(name="neptune_logger",     module=NeptuneLogger)
LOGGERS.register(name="NeptuneLogger",      module=NeptuneLogger)
LOGGERS.register(name="testtube",           module=TestTubeLogger)
LOGGERS.register(name="testtube_logger",    module=TestTubeLogger)
LOGGERS.register(name="TestTubeLogger",     module=TestTubeLogger)
