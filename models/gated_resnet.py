import torch.nn as nn

from models.layers import GatedResidualBLock, GatedConvLayer


class DGatedResNet(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [GatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels,
                                 dilation=1, normalize=False, local_condition=args.iso)]
        # Hidden layers
        for dilation in [2, 4, 8, 16, 1]:
            layers.append(GatedResidualBLock(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                             dilation=dilation, local_condition=args.iso))
        # Output layer
        layers.append(GatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, dilation=1,
                                     normalize=False, layer_activation=nn.Tanh(), local_condition=args.iso))
        self.model = nn.ModuleList(layers)

        self.residual = True

    def forward(self, x, c=None):
        out = x

        for layer in self.model:
            out = layer(out, c)

        if self.residual:   # learn noise residual
            out = out + x   # Should we apply tanh again after adding the residual?

        return out
