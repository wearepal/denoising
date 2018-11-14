import numpy as np
from numpy.random import RandomState

import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)

        # initialize weights to be as independent as possible using the He criterion
        weight_real, weight_im = _complex_he_init(self.conv_real.weight.shape)
        self.conv_real.weight.data = weight_real
        self.conv_im.weight.data = weight_im

        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            self.bias_im = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_im', None)

    def forward(self, x):
        """
        if x = a + ib, y = c + id, then:
        xy = (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        """
        x_real, x_im = torch.unbind(x, dim=-1)

        out_real = self.conv_real(x_real) - self.conv_im(x_im)
        out_im = self.conv_im(x_im) + self.conv_real(x_real)

        if self.bias:
            out_real += self.bias_real
            out_im += self.bias_im

        # Concatenate the real and imaginary values
        out = torch.cat([out_real[..., None], out_im[..., None]], dim=-1)
        return out


def _calculate_fan_in_and_fan_out(shape):
    """
    Copied with minor modifications from torch.nn.init
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if len(shape) > 2:
            receptive_field_size = np.prod(shape[1:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _complex_he_init(shape):
    """
    Initialize kernels to be independent from each other as much as possible.
    Derived from 'Deep Complex Networks' (https://arxiv.org/pdf/1705.09792.pdf).
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    s = 2. / fan_in

    rng = RandomState()
    modulus = rng.rayleigh(scale=s, size=shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)

    weight_real = modulus * np.cos(phase)
    weight_real = torch.from_numpy(weight_real).float()

    weight_im = modulus * np.sin(phase)
    weight_im = torch.from_numpy(weight_im).float()

    return weight_real, weight_im


def _test_complex_conv2d():
    x = torch.randn(2, 3, 12, 12)
    x_fourier = torch.rfft(x, onesided=True, signal_ndim=2)
    conv = ComplexConv2d(3, 3, kernel_size=3, padding=1)
    out = conv(x_fourier)
    out = torch.irfft(out, onesided=True, signal_ndim=2, signal_sizes=x.shape[2:])
    assert out.shape == x.shape
