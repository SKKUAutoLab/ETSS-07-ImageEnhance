#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NdLinear: next-gen replacement for ``nn.Linear``.

Reference:
    - https://github.com/ensemble-core/ndlinear
"""

__all__ = [
    "NdLinear",
]

import torch


class NdLinear(torch.nn.Module):
    """NdLinear: A PyTorch layer for projecting tensors into multi-space representations.
    
    Unlike conventional embedding layers that map into a single vector space, NdLinear
    transforms tensors across a collection of vector spaces, capturing multivariate
    structure and topical information that standard deep learning architectures
    typically lose.

    Args:
        input_dims: Shape of input tensor (excluding batch dimension).
        hidden_size: Target hidden dimensions after transformation.
    """
    
    def __init__(
        self,
        input_dims     : list | tuple,
        hidden_size    : list | tuple,
        transform_outer: bool = True
    ):
        super().__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims      = input_dims
        self.hidden_size     = hidden_size
        self.num_layers      = len(input_dims)  # Must match since dims are equal
        self.transform_outer = transform_outer

        # Define transformation layers per dimension
        self.align_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to project input tensor into a new multi-space representation.
        - Incrementally transposes, flattens, applies linear layers, and restores shape.

        Expected Input Shape: [batch_size, *input_dims]
        Output Shape: [batch_size, *hidden_size]

        Args:
            x: Input tensor with shape [batch_size, *input_dims]

        Returns:
            Output tensor with shape [batch_size, *hidden_size]
        """
        num_transforms = self.num_layers  # Number of transformations
        
        # Define iteration order
        # transform_indices = range(num_transforms) if transform_outer else reversed(range(num_transforms))
        
        for i in range(num_transforms):
            if self.transform_outer:
                layer         = self.align_layers[i]
                transpose_dim = i + 1
            else:
                layer         = self.align_layers[num_transforms - (i + 1)]
                transpose_dim = num_transforms - i

            # Transpose the selected dimension to the last position
            x = torch.transpose(x, transpose_dim, num_transforms).contiguous()

            # Store the original shape before transformation
            x_size = x.shape[:-1]

            # Flatten everything except the last dimension
            x = x.view(-1, x.shape[-1])

            # Apply transformation
            x = layer(x)
            
            # Reshape back to the original spatial structure (with new embedding dim)
            x = x.view(*x_size, x.shape[-1])

            # Transpose the dimension back to its original position
            x = torch.transpose(x, transpose_dim, num_transforms).contiguous()

        return x
