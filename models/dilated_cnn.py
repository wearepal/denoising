import torch.nn as nn

from models import GatedConvLayer


class DilatedGatedCNN(nn.Module):
    """
    Gated Conv Network with increasing dilation factor
    """
    def __init__(self, args):
        super().__init__()

        # Input + first hidden layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels, local_condition=args.iso)]
        # Hidden layers
        for d in [1, 2, 3, 4, 3, 2, 1]:
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
