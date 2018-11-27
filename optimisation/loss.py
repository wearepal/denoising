import torchvision
import torch.nn as nn
from torch.nn import MSELoss
from utils.metrics import *
from utils.functions import MeanShift


class VGGLoss(nn.Module):

    def __init__(self, args, rgb_range=2, prefactor=0.006):
        super().__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if args.vgg_feature_layer == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif args.vgg_feature_layer == '54':
            self.vgg = nn.Sequential(*modules[:35])

        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        # self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        self.prefactor = prefactor

    def forward(self, noisy, clean):
        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_noisy = _forward(noisy)
        with torch.no_grad():
            vgg_clean = _forward(clean.detach())

        loss = self.prefactor * F.mse_loss(vgg_noisy, vgg_clean)

        return loss


class WassersteinLossGAN(nn.Module):
    """
    Wasserstein (Earth mover's distance) loss for GAN discriminator
    """
    def __init__(self):
        super().__init__()

    def forward(self, fake, real, discriminator):
        """
        Args:
            fake: Fake samples produced by the generator
            real: Real (ground-truth) samples
            discriminator: Discriminator network

        Returns:
            Wasserstein loss for the discriminator
        """
        return discriminator(fake).mean() - discriminator(real).mean()


class HingeLossGAN(nn.Module):
    """
    Hinge loss for GAN discriminator
    """
    def __init__(self):
        super().__init__()

    def forward(self, fake, real, discriminator):
        """
        Args:
            fake: Fake samples produced by the generator
            real: Real (ground-truth) samples
            discriminator: Discriminator network

        Returns:
            Hinge loss for the discriminator
        """
        return (F.relu(1.0 - discriminator(real)).mean()) + (F.relu(1.0 + discriminator(fake)).mean())


class SobelMagnitude(nn.Module):

    def __init__(self, in_channels, out_channels=None, eps=1.e-5):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.eps = eps

        kernel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None][None]
        kernel_x = kernel_x.expand(out_channels, -1, -1, -1)
        self.sobel_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 bias=False, groups=in_channels)
        self.sobel_x.weight = nn.Parameter(kernel_x, requires_grad=False)

        kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None][None]
        kernel_y = kernel_y.expand(out_channels, -1, -1, -1)
        self.sobel_y = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 bias=False, groups=in_channels)
        self.sobel_y.weight = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x):
        G_x = self.sobel_x(x)
        G_y = self.sobel_y(x)
        G = ((G_x ** 2 + G_y ** 2) + self.eps).sqrt()   # channel-wise magnitude
        return G


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss function, where the pixels in the edges are granted higher weights
    compared to non-edge pixels (https://arxiv.org/pdf/1810.06766v1.pdf)
    """
    def __init__(self, in_channels=3, weight=0.025):
        super().__init__()
        self.weight = weight
        self.sobel = SobelMagnitude(in_channels)
        self.criterion = nn.MSELoss()

    def forward(self, noisy, clean):
        edge_map_noisy = self.sobel(noisy)
        edge_map_clean = self.sobel(clean)
        edge_loss = self.criterion(edge_map_noisy, edge_map_clean)
        return self.criterion(noisy, clean) + self.weight * edge_loss
