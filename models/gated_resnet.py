import torch.nn as nn

from models.layers import GatedResidualBLock, GatedConvLayer


class GatedResNet(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels,
                                 dilation=1, local_condition=args.iso)]
        # Hidden layers
        for _ in range((args.cnn_num_hidden_layers + 1) // 2):
            layers.append(GatedResidualBLock(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                             dilation=1, local_condition=args.iso))
        # Output layer
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, dilation=1,
                                     normalize=False, layer_activation=None, local_condition=args.iso))
        self.model = nn.ModuleList(layers)

        self.residual = True

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)

        if self.residual:   # learn noise residual
            out = out + x

        return out


class DGatedResNet(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels,
                                 dilation=1, local_condition=args.iso)]
        # Hidden layers
        for dilation in [2, 4, 8, 16, 1]:
            layers.append(GatedResidualBLock(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                             dilation=dilation, local_condition=args.iso))
        # Output layer
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, dilation=1,
                                     normalize=False, layer_activation=None, local_condition=args.iso))
        self.model = nn.ModuleList(layers)

        self.residual = True

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)

        if self.residual:   # learn noise residual
            out = out + x

        return out
