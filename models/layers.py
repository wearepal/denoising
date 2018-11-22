import math

import torch
import torch.nn as nn


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def norm(num_channels, num_norm_groups, num_classes=0):

    if num_classes > 1:
        return ConditionalNorm(num_features=num_channels, num_classes=num_classes,
                               num_groups=num_norm_groups)
    elif num_norm_groups > 0:
        return nn.GroupNorm(num_channels=num_channels, num_groups=num_norm_groups)
    else:
        return nn.BatchNorm2d(num_features=num_channels)


class ConvLayerParent(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 layer_activation=nn.ReLU(inplace=True), num_norm_groups=0, num_classes=0,
                 normalize=True, preserve_size=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if preserve_size:   # Does not work for stride > 1
            padding = int((((kernel_size + 1) / 2) - 1) * dilation)

        self.padding = padding
        self.dilation = dilation
        self.num_norm_groups = num_norm_groups
        self.conditional_norm = num_classes > 1
        self.normalize = normalize
        self.preserve_size = preserve_size

        self.conv: nn.Module
        self.norm: nn.Module
        self.layer_activation = layer_activation

    def forward(self, x, c=None, class_labels=None):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out, class_labels) if self.conditional_norm else self.norm(out)
        if self.layer_activation is not None:
            out = self.layer_activation(out)

        return out


class ConvLayer(ConvLayerParent):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 layer_activation=nn.ReLU(inplace=True), num_norm_groups=0, num_classes=0,
                 normalize=True, preserve_size=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         layer_activation, num_norm_groups, num_classes, normalize, preserve_size)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, bias=normalize)
        self.norm = norm(out_channels, num_norm_groups, num_classes) if normalize else None


class GatedConv2d(nn.Module):
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
            self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation)
            self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation)
            self.register_parameter('cond_features_bias', None)
            self.register_parameter('cond_gate_bias', None)
        else:
            # Because the conditioning data needs to be incorporated into the bias, bias is set to false
            self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                           self.padding, dilation, bias=False)
            self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                       self.padding, dilation, bias=False)
            self.cond_features_bias = nn.Parameter(torch.Tensor(out_channels))
            self.cond_gate_bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()

        if self.residual:
            self.downsample = None
            if stride != 1 or in_channels != out_channels:  # Maybe externalize this to GatedConvLayer
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


class GatedConvLayer(ConvLayerParent):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 conv_activation=None, layer_activation=nn.ReLU(inplace=True), local_condition=False,
                 conv_residual=True, num_norm_groups=0, num_classes=0,
                 normalize=True, preserve_size=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         layer_activation, num_norm_groups, num_classes, normalize, preserve_size)

        self.conv_activation = conv_activation
        self.local_condition = local_condition
        self.conv_residual = conv_residual

        self.conv = GatedConv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation,
                                conv_activation, local_condition, conv_residual)
        self.norm = norm(out_channels, num_norm_groups, num_classes) if normalize else None

    def forward(self, x, c=None, class_labels=None):
        out = self.conv(x, c)
        if self.norm is not None:
            out = self.norm(out, class_labels) if self.conditional_norm else self.norm(out)
        if self.layer_activation is not None:
            out = self.layer_activation(out)

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
            if stride != 1 or in_channels != out_channels:  # Maybe externalize
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
                 dilation=(1, 1), residual=True, num_norm_groups=0, num_classes=0):
        super().__init__()

        if isinstance(dilation, int):   # use the same dilation factor for both convolutions
            dilation = (dilation, dilation)

        conv1_padding = padding * dilation[0]    # ensure the dilation does not affect the output size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=conv1_padding, dilation=dilation[0], bias=False)
        self.norm1 = norm(out_channels, num_norm_groups, num_classes)

        conv2_padding = padding * dilation[1]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=conv2_padding, dilation=dilation[1], bias=False)
        self.norm2 = norm(out_channels, num_norm_groups, num_classes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm(num_norm_groups, out_channels)
            )

        self.stride = stride
        self.residual = residual
        self.conditional_norm = num_classes > 1

    def forward(self, x, class_labels=None):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out, class_labels) if self.conditional_norm else self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out, class_labels) if self.conditional_norm else self.norm2(out)

        if self.residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = self.relu(out)

        return out


class GatedResidualBLock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=(1, 1), residual=True, activation=None, local_condition=False,
                 block_residual=True, num_norm_groups=0, num_classes=0):
        super().__init__()

        if isinstance(dilation, int):   # use the same dilation factor for both convolutions
            dilation = (dilation, dilation)

        conv1_padding = padding * dilation[0]    # ensure the dilation does not affect the output size
        self.conv1 = GatedConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=conv1_padding, dilation=dilation[0], activation=activation,
                                 local_condition=local_condition, residual=block_residual)
        self.norm1 = norm(out_channels, num_norm_groups, num_classes)

        conv2_padding = padding * dilation[1]
        self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=conv2_padding, dilation=dilation[1], activation=activation,
                                 local_condition=local_condition, residual=block_residual)
        self.norm2 = norm(out_channels, num_norm_groups, num_classes)

        self.relu = nn.ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm(num_norm_groups, out_channels)
            )

        self.stride = stride
        self.residual = residual
        self.conditional_norm = num_classes > 1

    def forward(self, x, c=None, class_labels=None):
        residual = x

        out = self.conv1(x, c)
        out = self.norm1(out, class_labels) if self.conditional_norm else self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out, c)
        out = self.norm2(out, class_labels) if self.conditional_norm else self.norm2(out)

        if self.residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = self.relu(out)

        return out


class ConditionalNorm(nn.Module):

    def __init__(self, num_features, num_classes, num_groups=0):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_groups = num_groups

        if num_groups > 0:
            self.norm = nn.GroupNorm(num_channels=num_features, num_groups=num_groups, affine=False)
        else:
            self.norm = nn.BatchNorm2d(num_features, affine=False)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()    # Initialise bias at 0

    def forward(self, x, class_labels):
        out = self.norm(x)
        gamma, beta = self.embed(class_labels).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        return out


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out
