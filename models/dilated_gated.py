import torch.nn as nn

from models import GatedConvLayer


class DilatedGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels, local_condition=args.iso)]
        # Hidden layers
        for d in [2, 4, 8, 16]:
            layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                         dilation=d, preserve_size=True, local_condition=args.iso))
            layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                         dilation=d, preserve_size=True, local_condition=args.iso))
        # Output layer
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, local_condition=args.iso,
                                     normalize=False, layer_activation=None))
        self.model = nn.ModuleList(layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)

        if self.residual:   # learn noise residual
            out = out + x

        return out
