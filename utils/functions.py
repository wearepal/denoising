import torch.nn as nn
from torch.nn.utils import spectral_norm


def apply_spectral_norm(module):
    if hasattr(module, 'children'):
        children = list(module.children())
        for i in range(len(children)):
            if isinstance(children[i], (nn.Conv2d, nn.Linear)):
                children[i] = spectral_norm(children[i])
            if hasattr(children[i], 'children'):
                apply_spectral_norm(children[i])
