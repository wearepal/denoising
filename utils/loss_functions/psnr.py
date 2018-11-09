"""Compute the peak signal to noise ratio (PSNR)"""

import torch


class PSNR(torch.nn.Module):
    """Peak signal to noise ratio

    Infinity is the best value
    """
    def __init__(self, data_range):
        """
        Args:
            data_range: possible range of values. 255 for uint8; 1 for output of sigmoid
        """
        super().__init__()
        self.scale = 1.0 / data_range**2

    def forward(self, im_true, im_test):
        err = (im_true - im_test)**2
        mean_err = err.mean(-1).mean(-1).mean(-1)  # reduce the last three dimensions: HxWxD
        return -10 * torch.log10(self.scale * mean_err)
