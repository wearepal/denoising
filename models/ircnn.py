import torch.nn as nn

from models import ConvLayer


class IRNN(nn.Module):
    """
    Image restoration CNN (IRNN) from 'Learning Deep CNN Denoiser Prior for Image Restoration'
    https://arxiv.org/pdf/1704.03264.pdf
    """
    def __init__(self, args):
        super().__init__()

        # Input layer
        layers = [ConvLayer(args.cnn_in_channels, 64, normalize=True)]
        # Dilation factors for hidden layers
        dilation_factors = [2, 3, 4, 3, 2]
        # Hidden layers
        for dilation in dilation_factors:
            layers.append(ConvLayer(64, 64, dilation=dilation, preserve_size=True))
        # Output layer
        layers.append(ConvLayer(64, args.cnn_in_channels, normalize=False, layer_activation=None))
        self.model = nn.Sequential(*layers)

    def forward(self, x, c=None):
        noise_residual = self.model(x)
        denoised = x - noise_residual
        return denoised
