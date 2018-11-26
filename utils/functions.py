import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def apply_spectral_norm(module):
    if hasattr(module, 'children'):
        children = list(module.children())
        for i in range(len(children)):
            if isinstance(children[i], (nn.Conv2d, nn.Linear)):
                children[i] = spectral_norm(children[i])
            if hasattr(children[i], 'children'):
                apply_spectral_norm(children[i])
