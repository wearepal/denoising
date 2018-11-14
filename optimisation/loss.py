import torchvision
import torch.nn as nn
from torch.nn import MSELoss
from utils.metrics import *


class _FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super().__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class VGGLoss(nn.Module):
    """
    The VGG loss based on the ReLU activation layers of the
    pre-trained 19 layer VGG network. This is calculated as
    the euclidean distance between the feature representations
     of a reconstructed image.
    """
    feature_layer_default = 11  # VGG19 layer number from which to extract features

    def __init__(self, args=None, prefactor=0.006):
        """
        Args:
            prefactor: prefactor by which to scale the loss.
            Rescaling by a factor of 1 / 12.75 gives VGG losses of a scale that
            is comparable to MSE loss. This is equivalent to multiplying with a
            rescaling factor of ≈ 0.006.
        """
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True)
        self.criterion = nn.MSELoss()
        feature_layer = self.feature_layer_default if args is None else args.vgg_feature_layer
        self.feature_extractor = _FeatureExtractor(vgg, feature_layer=feature_layer)
        self.prefactor = prefactor

    def forward(self, fake, real):
        """
        Args:
            fake: Fake samples produced by the generator
            real: Real (ground-truth) samples
        Returns:
            VGG loss
        """
        real_features = self.feature_extractor(real).detach()
        fake_features = self.feature_extractor(fake)
        return self.prefactor * self.criterion(fake_features, real_features)


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
