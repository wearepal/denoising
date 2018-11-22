import torch
import torch.nn as nn

from models.layers import GatedConvLayer, ConvLayer


class SimpleCNN(nn.Module):
    """
    Simple generator network with a uniform number of filters
    throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [ConvLayer(args.cnn_in_channels, args.cnn_hidden_channels, num_classes=args.num_classes)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(ConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                    num_classes=args.num_classes))
        # Output layer
        layers.append(ConvLayer(args.cnn_hidden_channels, args.cnn_in_channels,
                                num_classes=args.num_classes, normalize=False,
                                layer_activation=None))
        # Output layer

        self.model = nn.Sequential(*layers)
        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = self.model(x)

        if self.residual:   # learn noise residual
            out = out + x

        return out


class SimpleGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels,
                                 local_condition=args.iso, num_classes=args.num_classes)]
        # # Hidden layers
        # for _ in range(args.cnn_num_hidden_layers):
        #     layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels,
        #                                  num_classes=args.num_classes, local_condition=args.iso))
        # Output layer
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels,
                                     num_classes=args.num_classes, local_condition=args.iso,
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
