import torch
import torch.nn as nn

from models import GatedConvLayer
from models.complex_layers import ComplexGatedConvLayer


class SimpleComplexGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [ComplexGatedConvLayer(args.cnn_in_channels, args.cnn_hidden_channels,
                                        local_condition=args.iso)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(ComplexGatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                                local_condition=args.iso))

        self.real_conv1 = GatedConvLayer(args.cnn_hidden_channels, args.cnn_hidden_channels * 2,
                                         local_condition=args.iso)
        self.real_conv2 = GatedConvLayer(args.cnn_hidden_channels * 2, args.cnn_in_channels,
                                         normalize=False, layer_activation=None, local_condition=args.iso)
        self.model = nn.ModuleList(layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = torch.rfft(x, signal_ndim=2)

        for layer in self.model:
            out = layer(out, c)

        out = torch.irfft(out, signal_ndim=2, signal_sizes=x.shape[2:])

        out = self.real_conv1(out, c)
        out = self.real_conv2(out, c)

        if self.residual:   # learn noise residual
            out = out + x
        # inverse fourier transform

        return out
