import torch.nn as nn


class BasicGenerator(nn.Module):
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
