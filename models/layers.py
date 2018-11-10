import math

import torch
import torch.nn as nn


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GatedConv2d(nn.Module):
    """
    Gated convolutional layer.
    Image ISO values can be used as extra conditioning data.
    If we want to use conditioning data that takes the form of a vector
    or coordinate map, a linear and convolutional layer, respectively would
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
            self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            self.register_parameter('cond_features_bias', None)
            self.register_parameter('cond_gate_bias', None)
        else:
            # Because the conditioning data needs to be incorporated into the bias, bias is set to false
            self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, bias=False)
            self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, bias=False)
            self.cond_features_bias = nn.Parameter(torch.Tensor(out_channels))
            self.cond_gate_bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()

        if self.residual:
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
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
            x: input tensor, [B, C, H, W]
            c: extra conditioning data (image ISO).
            In cases where c encodes spatial or sequential information (such as a sequence of linguistic features),
            the matrix products are replaced with convolutions.

        Returns:
            layer activations, hi
        """
        features = self.conv_features(x)
        gate = self.conv_gate(x)

        if self.local_condition and c is not None:
            c = c.view(-1, 1)   # enforce extra dimension for broadcasting
            features += (self.cond_features_bias * c).unsqueeze(-1).unsqueeze(-1)
            gate += (self.cond_gate_bias * c).unsqueeze(-1).unsqueeze(-1)

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


class GatedConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1,
                 activation=None, local_condition=False, residual=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.activation = activation
        self.local_condition = local_condition
        self.residual = residual

        self.sigmoid = nn.Sigmoid()

        if not local_condition:
            self.conv_features = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                    padding, output_padding, dilation=dilation)
            self.conv_gate = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                padding, output_padding, dilation=dilation)
            self.register_parameter('cond_features_bias', None)
            self.register_parameter('cond_gate_bias', None)
        else:
            # Because the conditioning data is incorporated into the bias, bias is set to false
            self.conv_features = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                    padding, output_padding, dilation=dilation, bias=False)
            self.conv_gate = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                padding, output_padding, dilation=dilation, bias=False)
            self.cond_features_bias = nn.Parameter(torch.Tensor(out_channels))
            self.cond_gate_bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()

        if self.residual:
            self.upsample = None
            if stride != 1 or in_channels != out_channels:
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride),
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
            x: input tensor, [B, C, H, W]
            c: extra conditioning data (image ISO)
            In cases where c encodes spatial or sequential information (such as a sequence of linguistic features),
            the matrix products are replaced with convolutions.

        Returns:
            layer activations, hi
        """
        features = self.conv_features(x)
        gate = self.conv_gate(x)

        if self.local_condition and c is not None:
            c = c.view(-1, 1)   # enforce extra dimension for broadcasting
            features += (self.cond_features_bias * c).unsqueeze(-1).unsqueeze(-1)
            gate += (self.cond_gate_bias * c).unsqueeze(-1).unsqueeze(-1)

        if self.activation is not None:
            features = self.activation(features)

        gate = self.sigmoid(gate)
        out = features * gate

        if self.residual:
            residual = x
            if self.upsample is not None:
                residual = self.upsample(residual)
            out += residual

        return out


class ResidualBLock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=(1, 1), residual=True):
        super().__init__()

        conv1_padding = padding * dilation[0]    # ensure the dilation does not affect the output size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=conv1_padding, dilation=dilation[0], bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        conv2_padding = padding * dilation[1]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=conv2_padding, dilation=dilation[1], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = self.relu(out)

        return out


class GatedResidualBLock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=(1, 1), residual=True, activation=None, local_condition=False,
                 block_residual=True):
        super().__init__()

        conv1_padding = padding * dilation[0]    # ensure the dilation does not affect the output size
        self.conv1 = GatedConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=conv1_padding, dilation=dilation[0], activation=activation,
                                 local_condition=local_condition, residual=block_residual)
        self.bn1 = nn.BatchNorm2d(out_channels)

        conv2_padding = padding * dilation[1]
        self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=conv2_padding, dilation=dilation[1], activation=activation,
                                 local_condition=local_condition, residual=block_residual)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.stride = stride
        self.residual = residual

    def forward(self, x, c=None):
        residual = x

        out = self.conv1(x, c)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, c)
        out = self.bn2(out)

        if self.residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = self.relu(out)

        return out


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()    # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        return out
