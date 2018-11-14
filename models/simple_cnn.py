import torch.nn as nn

from models.layers import GatedConv2d, GatedConvLayer, ConvLayer


class SimpleCNN(nn.Module):
    """
    Simple generator network with a uniform number of filters
    throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [ConvLayer(args.cnn_in_channels, args.cnn_hidden_channels)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(ConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels))
        # Output layer
        layers.append(nn.Conv2d(in_channels=args.cnn_hidden_channels, out_channels=args.cnn_in_channels,
                                kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self.tanh = nn.Tanh()
        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = self.tanh(self.model(x))

        if self.residual:   # learn noise residual
            out = out + x   # Should we apply tanh again after adding the residual?

        return out


class SimpleGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels, local_condition=args.iso)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels, local_condition=args.iso))
        # Output layer
        layers.append(GatedConv2d(args.cnn_hidden_channels, args.cnn_in_channels, local_condition=args.iso))
        self.model = nn.ModuleList(layers)
        self.tanh = nn.Tanh()

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)
        out = self.tanh(out)

        if self.residual:   # learn noise residual
            out = out + x   # Should we apply tanh again after adding the residual?

        return out
