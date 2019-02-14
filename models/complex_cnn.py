import torch
import torch.nn as nn

from models.complex_layers import ComplexConvLayer
from models.container import SequentialMeta
from models.fft import FourierTransformerReal


class ComplexNet(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        num_hidden_channels = args.cnn_hidden_channels // 2
        # Input layer
        layers = [ComplexConvLayer(args.cnn_in_channels, num_hidden_channels, num_classes=args.num_classes)]
        # Hidden layers
        for _ in range(args.cnn_hidden_layers):
            layers.append(ComplexConvLayer(num_hidden_channels, num_hidden_channels, num_classes=args.num_classes))

        layers.append(ComplexConvLayer(num_hidden_channels, args.cnn_in_channels, layer_activation=None, normalize=False, num_classes=args.num_classes))

        self.model = SequentialMeta(layers)
        self.transformer = FourierTransformerReal(signal_dim=2)

        self.residual = args.residual

    def forward(self, x, c=None, class_labels=None):
        out = self.transformer.transform(x)
        out = self.transformer.invert(self.model(out, c, class_labels))

        if self.residual:   # learn noise residual
            out = out + x

        return out
