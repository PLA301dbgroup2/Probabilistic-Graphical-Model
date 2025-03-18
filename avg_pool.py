# -*- coding: utf-8 -*-
"""
@File: max_pool.py

"""

import torch
from torch import nn

def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:

    if tensor.dim() != mask.dim():
        raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).bool(), replace_with)


class AvgPoolerAggregator(nn.Module):

    def __init__(self, ) -> None:
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  
        
        if mask is not None:
            input_tensors = replace_masked_values(
                input_tensors, mask.unsqueeze(2), 0)

        tokens_avg_pooled = torch.mean(input_tensors, 1)

        return tokens_avg_pooled
