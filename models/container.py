import torch.nn as nn
from inspect import signature
import torch
from models.layers import ConvLayer


class SequentialMeta(nn.Module):
    """
    A generalized nn.Sequential container allowing metadata to be passed to successive layers
    """
    def __init__(self, layer_list):
        super(SequentialMeta, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, *args):
        for layer in self.chain:
            x = layer(x, args)
        return x

