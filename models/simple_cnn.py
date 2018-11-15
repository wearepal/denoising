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
        layers.append(ConvLayer(args.cnn_hidden_channels, args.cnn_in_channels,
                                normalize=False, layer_activation=None))
        # Output layer

        self.model = nn.Sequential(*layers)
        self.tanh = nn.Tanh()
        self.residual = not args.interpolate

    def forward(self, x, c=None):
        out = self.model(x)

        if self.residual:   # learn noise residual
            out = out + x   # Should we apply tanh again after adding the residual?
        out = self.tanh(out)

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
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, local_condition=args.iso,
                                     normalize=False, layer_activation=None))
        self.model = nn.ModuleList(layers)
        self.tanh = nn.Tanh()

        self.residual = not args.interpolate

    def forward(self, x, c=None):
        out = x

        for layer in self.model:
            out = layer(out, c)

        if self.residual:   # learn noise residual
            out = out + x   # Should we apply tanh again after adding the residual?
        out = self.tanh(out)

        return out
