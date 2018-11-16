import torch
import torch.nn as nn

from models import GatedConvLayer
from models.complex_layers import ComplexGatedConvLayer


class HybridGatedCNN(nn.Module):
    """
    Hybrid dual-stream (one real, one complex) gated CNN
    """
    def __init__(self, args, stream_lenth=3):
        super().__init__()

        num_real_filters = args.cnn_hidden_channels
        num_complex_filters = args.cnn_hidden_channels // 2

        # Real stream
        real_layers = [GatedConvLayer(args.cnn_in_channels, num_real_filters)]
        for i in range(stream_lenth):
            dilation = 2 * (i + 1)  # double dilation factor each layer
            real_layer = GatedConvLayer(num_real_filters, num_real_filters,
                                        local_condition=args.iso, dilation=dilation,
                                        preserve_size=True)
            real_layers.append(real_layer)

        # Complex stream
        complex_layers = [ComplexGatedConvLayer(args.cnn_in_channels, num_complex_filters,
                                                local_condition=args.iso)]
        for i in range(stream_lenth):
            dilation = 2 * (i + 1)  # double dilation factor each layer
            complex_layer = ComplexGatedConvLayer(num_complex_filters, num_complex_filters,
                                                  local_condition=args.iso, dilation=dilation,
                                                  preserve_size=True)
            complex_layers.append(complex_layer)

        # Combine outputs of complex and real streams
        self.pooling_conv = GatedConvLayer(num_real_filters + num_complex_filters, num_real_filters,
                                           local_condition=args.iso)
        self.output_conv = GatedConvLayer(num_real_filters, args.cnn_in_channels,
                                          normalize=False, layer_activation=None,
                                          local_condition=args.iso)

        self.complex_stream = nn.ModuleList(complex_layers)
        self.real_stream = nn.ModuleList(complex_layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):

        # Complex stream
        complex_out = torch.rfft(x, signal_ndim=2)
        for layer in self.complex_stream:
            complex_out = layer(complex_out, c)
        complex_out = torch.irfft(complex_out, signal_ndim=2, signal_sizes=x.shape[2:])

        # Real stream
        real_out = x
        for layer in self.real_stream:
            real_out = layer(real_out, c)

        # Concatenate outputs of real and complex streams
        cat_streams = torch.cat([complex_out, real_out], dim=1)
        # Convolve over combined output
        out = self.pooling_conv(cat_streams, c)
        out = self.output_conv(out, c)

        if self.residual:   # learn noise residual
            out = out + x

        return out
