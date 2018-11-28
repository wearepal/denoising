import torch
import torch.nn as nn

from models import ConvLayer, GatedConvLayer


class ResidualDenseBlock(nn.Module):

    def __init__(self, nc, gc=32, kernel_size=3, local_condition=True, beta=0.2):
        super().__init__()
        self.beta = beta

        self.conv1 = ConvLayer(nc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv2 = ConvLayer(nc+gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv3 = ConvLayer(nc+2*gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.conv4 = ConvLayer(nc+3*gc, gc, kernel_size=kernel_size, normalize=False,
                               layer_activation=nn.LeakyReLU(0.2))
        self.gated_conv = GatedConvLayer(nc+4*gc, gc, kernel_size=kernel_size,
                                         local_condition=local_condition, conv_residual=False,
                                         normalize=False, layer_activation=None)

    def forward(self, x, c=None, class_labels=None):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        out = self.gated_conv(torch.cat((x, x1, x2, x3, x4), 1), c)
        return out.mul(self.beta) + x


class RDDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, local_condition=True, beta=0.2):
        super().__init__()
        self.beta = beta
        self.RDB1 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, beta)
        self.RDB2 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, beta)
        self.RDB3 = ResidualDenseBlock(nc, gc, kernel_size, local_condition, beta)

    def forward(self, x, c=None, class_labels=None):
        out = self.RDB1(x, c, class_labels)
        out = self.RDB2(out, c, class_labels)
        out = self.RDB3(out, c, class_labels)
        return out.mul(self.beta) + x


class DenseGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [ConvLayer(args.cnn_in_channels, 32, normalize=False, layer_activation=None)]
        # Hidden layers
        for _ in range(4):
            layers.append(RDDB(32, 32, local_condition=args.iso, beta=0.4))
        # Output layer
        layers.append(ConvLayer(32, args.cnn_in_channels, normalize=False, layer_activation=None))
        self.model = nn.ModuleList(layers)

        # init
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                m.weight.data *= 0.1

        self.apply(init_weights)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):
        out = x

        for layer in self.model:
            out = layer(out, c, class_labels)

        if self.residual:   # learn noise residual
            out = out + x

        return out
