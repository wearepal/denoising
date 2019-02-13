import torch
import torch.nn as nn

from models import GatedConvLayer
from models.complex_layers import ComplexGatedConvLayer


class HybridGatedCNN(nn.Module):
    """
    Hybrid dual-stream (one real, one complex) gated CNN
    """
    def __init__(self, args, stream_length=3):
        super().__init__()

        num_real_filters = args.cnn_hidden_channels
        num_complex_filters = args.cnn_hidden_channels // 2

        # Real stream
        real_layers = [GatedConvLayer(args.cnn_in_channels, num_real_filters)]
        for i in range(stream_length):
            dilation = 2 ** (i + 1)
            # dilation = 1
            real_layer = GatedConvLayer(num_real_filters, num_real_filters,
                                        local_condition=args.iso, dilation=dilation,
                                        preserve_size=True)
            real_layers.append(real_layer)

        # Complex stream
        complex_layers = [ComplexGatedConvLayer(args.cnn_in_channels, num_complex_filters,
                                                local_condition=args.iso)]
        for i in range(stream_length):
            dilation = 2 ** (i + 1)  # double dilation factor each layer
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

        self.real_stream = nn.ModuleList(real_layers)
        self.complex_stream = nn.ModuleList(complex_layers)

        self.residual = args.residual

    def forward(self, x, c=None, class_labels=None):

        # Real stream
        real_out = x
        for r_layer in self.real_stream:
            real_out = r_layer(real_out, c)

        # Complex stream
        complex_out = torch.rfft(x, signal_ndim=2)
        for c_layer in self.complex_stream:
            complex_out = c_layer(complex_out, c)
        complex_out = torch.irfft(complex_out, signal_ndim=2, signal_sizes=x.shape[2:])

        # Concatenate outputs of real and complex streams
        cat_streams = torch.cat([complex_out, real_out], dim=1)
        # Convolve over combined output
        out = self.pooling_conv(cat_streams, c)
        out = self.output_conv(out, c)

        if self.residual:   # learn noise residual
            out = out + x

        return out
