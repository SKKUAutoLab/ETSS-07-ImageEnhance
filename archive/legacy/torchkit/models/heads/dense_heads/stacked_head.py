# ==================================================================== #
# File name: stacked_head.py
# Author: Long H. Pham
# Date created: 08/30/2021
# The `torchkit.models.heads.dense_heads.stacked_head` defines the
# classifier head with several hidden fc layers and a output fc layer.
# ==================================================================== #
from __future__ import annotations

import logging
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchkit.core.layer import Module
from torchkit.core.layer import ModuleList
from torchkit.core.utils import ForwardXYOutput
from torchkit.core.utils import InputTensor
from torchkit.core.utils import OutputTensor
from torchkit.models.builder import ACT_LAYERS
from torchkit.models.builder import HEADS
from torchkit.models.builder import NORM_LAYERS
from .cls_head import ClsHead

logger = logging.getLogger()


# MARK: - LinearBlock

class LinearBlock(Module):
    """Implement a linear block with a series of fc/norm/act/dropout layers.
    
    Attributes:
        fc (nn.Module):
            Fully-connected layer.
        norm (nn.Module):
            Normalize layer.
        act (nn.Module):
            Activation layer.
        dropout (nn.Module):
            Dropout layer.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout_rate: float          = 0.0,
        norm_cfg    : Optional[dict] = None,
        act_cfg     : Optional[dict] = None,
        init_cfg    : Optional[dict] = None,
        *args, **kwargs
    ):
        """
        
        Args:
            in_channels (int):
                Number of channels in the input feature map.
            out_channels (int):
                Number of channels in the output feature map.
            dropout_rate (float):
                The dropout rate.
            norm_cfg (dict, optional):
                The normalize layer's config. Default: `None`.
            act_cfg (dict, optional):
                The activation layer's config. Default: `None`.
            init_cfg (dict, optional):
                The extra init config of layers. Default: `None`.
        """
        super().__init__(init_cfg=init_cfg, *args, **kwargs)
        self.fc      = nn.Linear(in_channels, out_channels)
        self.norm    = None
        self.act     = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = NORM_LAYERS.build_from_cfg(cfg=norm_cfg, num_features=out_channels)[1]
        if act_cfg is not None:
            self.act = ACT_LAYERS(cfg=act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    # MARK: Forward Pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.
        
        Args:
            x (torch.Tensor):
                The input tensor.

        Returns:
             x (torch.Tensor):
                The output tensor.
        """
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# noinspection PyDefaultArgument
@HEADS.register(name="stacked_linear_cls_head")
@HEADS.register(name="StackedLinearClsHead")
class StackedLinearClsHead(ClsHead):
    """Classifier head with several hidden fc layer and an output fc layer.
    
    Attributes:
        num_classes (int):
            Number of categories excluding the background category.
        in_channels (int):
            Number of channels in the input feature map.
        mid_channels (Sequence):
            Number of channels in the hidden fc layers.
        dropout_rate (float):
            Dropout rate after each hidden fc layer, except the last layer. Default: `0.0`.
        norm_cfg (dict, optional):
            Config dict of normalization layer after each hidden fc layer, except the last layer. Default: `None`.
        act_cfg (dict, optional):
            Config dict of activation function after each hidden layer, except the last layer.
            Default: `dict(name="ReLU")`.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        num_classes : int,
        in_channels : int,
        mid_channels: Sequence,
        dropout_rate: float         = 0.0,
        norm_cfg    : dict          = None,
        act_cfg     : dict          = dict(name="ReLU"),
        name        : Optional[str] = "stacked_linear_cls_head",
        *args, **kwargs
    ):
        """
        
        Args:
            num_classes (int):
                Number of categories excluding the background category.
            in_channels (int):
                Number of channels in the input feature map.
            mid_channels (Sequence):
                Number of channels in the hidden fc layers.
            dropout_rate (float):
                Dropout rate after each hidden fc layer, except the last layer. Default: `0.0`.
            norm_cfg (dict, optional):
                Config dict of normalization layer after each hidden fc layer, except the last layer. Default: `None`.
            act_cfg (dict, optional):
                Config dict of activation function after each hidden layer, except the last layer.
                Default: `dict(name="ReLU")`.
			name (str, optional):
				Name of the head. Default: `stacked_linear_cls_head`.
        """
        super().__init__(
            name = name,
            *args, **kwargs
        )
        assert num_classes > 0, \
            f"`num_classes` must be a positive integer, got {num_classes} instead."
        assert isinstance(mid_channels, Sequence), \
            f"`mid_channels` should be a sequence, instead of {type(mid_channels)}."
        self.num_classes  = num_classes
        self.in_channels  = in_channels
        self.mid_channels = mid_channels
        self.dropout_rate = dropout_rate
        self.norm_cfg     = norm_cfg
        self.act_cfg      = act_cfg

        self._init_layers()

    # MARK: Configure
    
    def _init_layers(self):
        self.layers = ModuleList(init_cfg=dict(name="Normal", layer="Linear", mean=0.0, std=0.01, bias=0.0))
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels  = in_channels,
                    out_channels = hidden_channels,
                    dropout_rate = self.dropout_rate,
                    norm_cfg     = self.norm_cfg,
                    act_cfg      = self.act_cfg
                )
            )
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                in_channels  = self.mid_channels[-1],
                out_channels = self.num_classes,
                dropout_rate = 0.0,
                norm_cfg     = None,
                act_cfg      = None
            )
        )

    def init_weights(self):
        self.layers.init_weights()

    # MARK: Forward Pass

    def forward_xy(self, x: InputTensor, y: InputTensor, *args, **kwargs) -> ForwardXYOutput:
        """Classifier head with several hidden fc layer and an output fc layer. Both `x` and `y` are given, hence, we
        compute the loss and metrics also.

		Args:
			x (InputTensor):
				`x` contains either the input data or the predictions from previous step.
			y (InputTensor):
				`y` contains the ground truth.

		Returns:
			(ForwardXYOutput):
				y_hat (OutputTensor):
					The final predictions tensor.
				metrics (MetricData):
					- A dictionary with the first key must be the `loss`.
					- `None`, training will skip to the next batch.
		"""
        if isinstance(x, tuple):
            x = x[-1]
        y_hat = x
        for layer in self.layers:
            y_hat = layer(y_hat)
            
        # NOTE: Calculate loss and metrics from logits
        metrics = self.loss_metrics(y_hat=y_hat, y=y)
    
        # NOTE: Calculate class-score (softmax)
        y_hat = F.softmax(y_hat, dim=1) if y_hat is not None else None

        return y_hat, metrics

    def forward_x(self, x: InputTensor, *args, **kwargs) -> OutputTensor:
        """Classification head for multilabel task. During inference, only `x` is given so we compute `y_hat` only.

		Args:
			x (InputTensor):
				`x` contains either the input data or the predictions from previous step.

		Returns:
			y_hat (OutputTensor):
				The final prediction.
		"""
        if isinstance(x, tuple):
            x = x[-1]
        y_hat = x
        for layer in self.layers:
            y_hat = layer(y_hat)
        if isinstance(y_hat, list):
            y_hat = sum(y_hat) / float(len(y_hat))
    
        # NOTE: Calculate class-score (softmax)
        y_hat = F.softmax(y_hat, dim=1) if y_hat is not None else None

        return y_hat
