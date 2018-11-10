import torch.nn as nn

from models.layers import GatedConv2d


class SimpleCNN(nn.Module):
    """
    Simple generator network with a uniform number of filters
    throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        def _generator_block(_in_channels, _out_channels):
            conv = nn.Conv2d(_in_channels, _out_channels, kernel_size=3, stride=1, padding=1)
            block = nn.Sequential(
                conv,
                nn.BatchNorm2d(_out_channels),
                nn.ReLU()
            )
            return block

        # Input layer
        layers = [_generator_block(args.cnn_in_channels, args.cnn_hidden_channels)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(_generator_block(args.cnn_hidden_channels, args.cnn_hidden_channels))
        # Output layer
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=args.cnn_hidden_channels, out_channels=args.cnn_in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x, c=None):
        return self.model(x)


class SimpleGatedCNN(nn.Module):
    """
    Simple generator network with gated convolutions and a
    uniform number of filters throughout the hidden layers.
    """
    def __init__(self, args):
        super().__init__()

        class _GeneratorBlock(nn.Module):
            def __init__(self, _in_channels, _out_channels, _local_condition):
                super().__init__()
                self.conv = GatedConv2d(_in_channels, _out_channels, kernel_size=3, stride=1,
                                        padding=1, local_condition=_local_condition)
                self.bn = nn.BatchNorm2d(_out_channels)

            def forward(self, x, c):
                return self.bn(self.conv(x, c))

        # Input layer
        layers = [_GeneratorBlock(args.cnn_in_channels, args.cnn_hidden_channels,
                                  _local_condition=args.iso)]
        # Hidden layers
        for _ in range(args.cnn_num_hidden_layers):
            layers.append(_GeneratorBlock(args.cnn_hidden_channels, args.cnn_hidden_channels,
                                          _local_condition=args.iso))
        # Output layer
        layers.append(GatedConv2d(args.cnn_hidden_channels, args.cnn_in_channels,
                                  local_condition=args.iso))
        self.model = nn.ModuleList(layers)
        self.tanh = nn.Tanh()

    def forward(self, x, c=None):
        out = x
        for layer in self.model:
            out = layer(out, c)

        return self.tanh(out)
