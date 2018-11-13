import torch.nn as nn
from torch.nn import functional as F
import torch


class Smoothing(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=2)
        self.deconv1 = nn.ConvTranspose2d(3, 16, kernel_size=5, stride=2)
        self.merge_conv = nn.Conv2d(6, 3, kernel_size=5, stride=1)

    def forward(self, x):
        # downsampling
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        # upsampling to get to original size
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv1(out))
        # stack the learned "smoothing weights" and the original image
        stacked = torch.cat([out, x], dim=1)
        # apply basically a weighted smoothing to the input
        out = self.merge_conv(stacked)

        return out
