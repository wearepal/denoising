import torch


class FourierTransformerReal:

    def __init__(self, signal_dim=2):
        self.signal_dim = signal_dim

    def transform(self, x):
        return torch.rfft(x, signal_ndim=self.signal_dim)

    def invert(self, x):
        return torch.irfft(x, signal_ndim=self.signal_dim, signal_sizes=x.shape[2:])
