import torch
import torch.nn as nn

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
        # Output layer
        layers.append(ComplexGatedConvLayer(args.cnn_hidden_channels, args.cnn_in_channels, local_condition=args.iso,
                                            normalize=False, layer_activation=None))
        self.model = nn.ModuleList(layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        fourier_x = torch.rfft(x, signal_ndim=2)

        out = fourier_x
        for layer in self.model:
            out = layer(out, c)

        if self.residual:   # learn noise residual
            out = out + fourier_x
        # inverse fourier transform
        out = torch.irfft(out, signal_ndim=2, signal_sizes=x.shape[2:])
        out = out.tanh()

        return out
