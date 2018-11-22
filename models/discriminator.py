import numpy as np

import torch.nn as nn

from models.layers import GatedConvLayer

import argparse

PATCH_SIZE = 64


class GatedDiscriminator(nn.Module):

    def __init__(self, args):
        super().__init__()

        layers = [GatedConvLayer(args.cnn_in_channels, 32, stride=2, normalize=False,
                                 layer_activation=nn.LeakyReLU(0.1), local_condition=args.iso,
                                 num_classes=args.num_classes)]

        in_channels = 32
        in_dim = PATCH_SIZE / layers[0].stride      # patch size is 64

        # layer_params = [(32, 4, 2), (64, 3, 1), (64, 4, 2), (128, 3, 1),
        #                 (128, 4, 2), (256, 3, 1), (256, 4, 2), (512, 3, 1)]
        layer_params = [(64, 3, 2), (128, 3, 2), (256, 3, 1), (512, 3, 2)]

        for out_channels, kernel_size, stride in layer_params:
            conv = GatedConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  normalize=False, layer_activation=nn.LeakyReLU(0.1),
                                  local_condition=args.iso, num_classes=args.num_classes)
            layers.append(conv)
            in_channels = out_channels
            in_dim /= stride

        self.model = nn.ModuleList(layers)
        self.fc = nn.Linear(in_dim * in_dim * in_channels, 1)

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)
            print(out.shape)

        out = self.fc(out.view(x.size(0), -1))

        return out
