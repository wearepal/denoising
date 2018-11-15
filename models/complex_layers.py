import math

import numpy as np
from numpy.random import RandomState

import torch
import torch.nn as nn

from models.layers import ConvLayerParent


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=False)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, bias=False)

        # initialize weights to be as independent as possible using the He criterion
        weight_real, weight_im = _complex_he_init(self.conv_real.weight.shape)
        self.conv_real.weight.data = weight_real
        self.conv_im.weight.data = weight_im

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 2))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        if x = a + ib, y = c + id, then:
        xy = (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        """
        x_real, x_im = torch.unbind(x, dim=-1)

        out_real = self.conv_real(x_real) - self.conv_im(x_im)
        out_im = self.conv_im(x_real) + self.conv_real(x_im)
        # Concatenate the real and imaginary values to form a complex matrix
        out = torch.cat([out_real[..., None], out_im[..., None]], dim=-1)

        if self.bias is not None:
            out += self.bias

        return out


class ComplexConvLayer(ConvLayerParent):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 layer_activation=nn.ReLU(inplace=True), num_norm_groups=0, num_classes=0,
                 normalize=True, preserve_size=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         layer_activation, num_norm_groups, num_classes, normalize, preserve_size)

        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=normalize)
        self.norm = ComplexBatchNorm2d(out_channels) if normalize else None


class ComplexGatedConv2d(nn.Module):
    """
    Gated convolutional layer.
    Image ISO values can be used as extra conditioning data.
    If we want to use conditioning data that takes the form of a vector
    or coordinate map, a linear and convolutional layer, respectively, would
    need to be used for the biases.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 activation=None, local_condition=False, residual=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation
        self.local_condition = local_condition
        self.residual = residual

        self.sigmoid = nn.Sigmoid()

        if not local_condition:
            self.conv_features = ComplexConv2d(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation)
            self.conv_gate = ComplexConv2d(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation)
            self.register_parameter('cond_features_bias', None)
            self.register_parameter('cond_gate_bias', None)
        else:
            # Because the conditioning data needs to be incorporated into the bias, bias is set to false
            self.conv_features = ComplexConv2d(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, bias=False)
            self.conv_gate = ComplexConv2d(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, bias=False)
            self.cond_features_bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 2))
            self.cond_gate_bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 2))
            self.reset_parameters()

        if self.residual:
            self.downsample = None
            if stride != 1 or in_channels != out_channels:  # Maybe externalize this to GatedConvLayer
                self.downsample = nn.Sequential(
                    ComplexConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                )

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels)
        if self.cond_features_bias is not None:
            self.cond_features_bias.data.uniform_(-stdv, stdv)
        if self.cond_gate_bias is not None:
            self.cond_gate_bias.data.uniform_(-stdv, stdv)

    def forward(self, x, c=None):
        """
        hi = σ(Wg,i ∗ xi + V^T g,ic) * activation(Wf,i ∗ xi + V^Tf,ic)

        Args:
            x: input tensor, [B, C, H, W, 2]
            c: extra conditioning data (image ISO).
            In cases where c encodes spatial or sequential information (such as a sequence of linguistic features),
            the matrix products are replaced with convolutions.

        Returns:
            layer activations, hi
        """
        features = self.conv_features(x)
        gate = self.conv_gate(x)

        if self.local_condition and c is not None:
            c = c.view(-1, 1, 1, 1, 1)
            features += (self.cond_features_bias * c)
            gate += (self.cond_gate_bias * c)

        if self.activation is not None:
            features = self.activation(features)

        gate = self.sigmoid(gate)
        out = features * gate

        if self.residual:
            residual = x
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        return out


class ComplexGatedConvLayer(ConvLayerParent):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 conv_activation=None, layer_activation=nn.ReLU(inplace=True), local_condition=False,
                 conv_residual=True, num_norm_groups=0, num_classes=0,
                 normalize=True, preserve_size=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         layer_activation, num_norm_groups, num_classes, normalize, preserve_size)

        self.conv_activation = conv_activation
        self.local_condition = local_condition
        self.conv_residual = conv_residual

        self.conv = ComplexGatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                       conv_activation, local_condition, conv_residual)
        self.norm = ComplexBatchNorm2d(out_channels) if normalize else None

    def forward(self, x, c=None, class_labels=None):
        out = self.conv(x, c)
        if self.norm is not None:
            out = self.norm(out, class_labels) if self.conditional_norm else self.norm(out)
        if self.layer_activation is not None:
            out = self.layer_activation(out)

        return out


class ComplexBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True,
                 track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            # scaling
            gamma_shape = (1, num_features, 1, 1)
            self.gamma_rr = nn.Parameter(torch.full(gamma_shape, fill_value=1. / math.sqrt(2)))
            self.gamma_ii = nn.Parameter(torch.full(gamma_shape, fill_value=1. / math.sqrt(2)))
            self.gamma_ri = nn.Parameter(torch.zeros(gamma_shape))
            # centering
            beta_shape = gamma_shape + (2,)
            self.beta = nn.Parameter(torch.zeros(beta_shape))
        else:
            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ii', None)
            self.register_parameter('gamma_ri', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_Vrr', torch.full((num_features,),
                                                           fill_value=1. / math.sqrt(2)))
            self.register_buffer('running_Vii', torch.full((num_features,),
                                                           fill_value=1. / math.sqrt(2)))
            self.register_buffer('running_Vri', torch.zeros(num_features))
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_im', torch.zeros(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_Vrr', None)
            self.register_buffer('running_Vii', None)
            self.register_buffer('running_Vri', None)
            self.register_buffer('running_mean_real', None)
            self.register_buffer('running_mean_im', None)
            self.register_buffer('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_Vrr.fill_(1. / math.sqrt(2))
            self.running_Vri.fill_(1. / math.sqrt(2))
            self.running_Vii.zero_()
            self.running_mean_real.zero_()
            self.running_mean_im.zero_()
            self.num_batches_tracked.zero_()

    @staticmethod
    def _moving_average(new_value, running_mean, momentum):
        return momentum * new_value + (1 - momentum) * running_mean

    def _update_stats(self, Vrr, Vii, Vri, mean_real, mean_im, momentum):
        if self.affine:
            self.running_Vrr = self._moving_average(Vrr, self.running_Vrr, momentum)
            self.running_Vii = self._moving_average(Vii, self.running_Vii, momentum)
            self.running_Vri = self._moving_average(Vri, self.running_Vri, momentum)
            self.running_mean_real = self._moving_average(mean_real, self.running_mean_real, momentum)
            self.running_mean_im = self._moving_average(mean_im, self.running_mean_im, momentum)

    @staticmethod
    def _whiten(centered_real, centered_im, Vrr, Vii, Vri):
        """
        The normalization procedure allows one to decorrelate the
        imaginary and real parts of a unit
        """
        # We require the covariance matrix's inverse square root. That first requires
        # square rooting, followed by inversion

        # trace = Vrr + Vii. Guaranteed >= 0 because SPD.
        trace = Vrr + Vii
        # determinant. Guaranteed >= 0 because SPD
        det = (Vrr * Vii) - (Vri ** 2)

        s = det.sqrt()  # determinant of square root matrix
        t = (trace + 2 * s).sqrt()

        # # The square root matrix could now be explicitly formed as
        #             [ Vrr+s Vri   ]
        #       (1/t) [ Vir   Vii+s ]
        #  # but we don't need to do this immediately since we can also simultaneously
        #  invert. We can do this because we've already computed the determinant of
        #  the square root matrix, and can thus invert it using the analytical
        #  solution for 2x2 matrices.
        #             [  Vii+s  -Vri   ]
        #   (1/s)(1/t)[ -Vir     Vrr+s ]
        inverse_st = 1. / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st

        out_real = Wrr * centered_real + Wri * centered_im
        out_im = Wri * centered_real + Wii * centered_im

        return out_real, out_im

    def complex_batch_norm(self, x, training=False, momentum=0.9):
        x_real, x_im = torch.unbind(x, dim=-1)

        if training:
            x_real_mean = x_real.mean()
            x_im_mean = x_im.mean()
            centered_real = x_real - x_real_mean
            centered_im = x_im - x_im_mean
            # Covariance matrices
            Vrr = (centered_real ** 2).mean() + self.eps
            Vii = (centered_im ** 2).mean() + self.eps
            Vri = (centered_real * centered_im).mean()
            # update moving averages
            self._update_stats(Vrr, Vii, Vri, x_real_mean, x_im_mean, momentum)
        else:
            centered_real = x_real - self.running_mean_real
            centered_im = x_im - self.running_mean_im
            Vrr = self.running_Vrr
            Vii = self.running_Vii
            Vri = self.running_Vri

        out_real, out_im = self._whiten(centered_real, centered_im, Vrr, Vii, Vri)

        if self.affine:
            out_real = (self.gamma_rr * out_real + self.gamma_ri * out_im)
            out_im = (self.gamma_ri * out_real + self.gamma_ii * out_im)
            out = torch.cat([out_real[..., None], out_im[..., None]], dim=-1) + self.beta
        else:
            out = torch.cat([out_real[..., None], out_im[..., None]], dim=-1)

        return out

    def forward(self, x):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return self.complex_batch_norm(x, self.training or not self.track_running_stats,
                                       exponential_average_factor)


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
    x = torch.randn(6, 3, 12, 12)
    x_fourier = torch.rfft(x, onesided=True, signal_ndim=2)
    conv = ComplexConv2d(3, x.size(1), kernel_size=3, padding=1)
    out = conv(x_fourier)
    out = torch.irfft(out, onesided=True, signal_ndim=2, signal_sizes=x.shape[2:])
    assert out.shape == x.shape


def _test_complex_gated_conv2d():
    x = torch.randn(6, 3, 12, 12)
    x_fourier = torch.rfft(x, onesided=True, signal_ndim=2)
    c = torch.randn(x.size(0))
    conv = ComplexGatedConv2d(x.size(1), 3, kernel_size=3, padding=1, local_condition=True)
    out = conv(x_fourier, c)
    out = torch.irfft(out, onesided=True, signal_ndim=2, signal_sizes=x.shape[2:])
    assert out.shape == x.shape


def _test_complex_batch_norm():
    x = torch.randn(6, 3, 12, 12, 2)
    bn = ComplexBatchNorm2d(3)
    out = bn(x)
    assert out.shape == x.shape
