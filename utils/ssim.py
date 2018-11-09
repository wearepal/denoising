"""
PyTorch differentiable Structural Similarity (SSIM)

MIT licensed

originally from https://github.com/jorge-pessoa/pytorch-msssim
"""
import torch
import torch.nn.functional as F
import numpy as np


def _gaussian(window_size, sigma):
    gauss = torch.Tensor(np.exp(-(np.arange(window_size) - window_size // 2)**2 / (2.0 * sigma**2)))
    return gauss / gauss.sum()


def _create_window(window_size, channel=1):
    window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, data_range, window_size=11, window=None, full=False):
    """Compute Structural Similarity"""
    padd = 0  # padding for convolution
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = _create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        cs = torch.mean(v1 / v2)  # contrast sensitivity
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    """Structural Similarity index

    1 is the best value
    """
    def __init__(self, data_range, channels, window_size=11):
        """
        Attention: assumes the structure NCHW ("channels first") for the images; not NHWC

        Args:
            data_range: the range of the expected values in the images. usually 255 (for uint8).
                        for output of sigmoid it would be 1.0; for output of tanh it would be 2.0
            channels: number of color channels in the images
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

        # Assume 1 channel for SSIM
        self.channels = channels
        # Re-use window
        self.window = _create_window(self.window_size, channels)

    def forward(self, img1, img2):
        if self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self.window.to(img1.device).type(img1.dtype)

        return ssim(img1, img2, self.data_range, window=self.window, window_size=self.window_size)
