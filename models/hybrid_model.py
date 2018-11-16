import torch
import torch.nn as nn

from models import GatedConvLayer
from models.complex_layers import ComplexGatedConvLayer


class HybridGatedCNN(nn.Module):
    """
    Hybrid dual-stream (one real, one complex) gated CNN
    """
    def __init__(self, args, stream_length=3):
        super().__init__()

        num_real_filters = args.cnn_hidden_channels
        num_complex_filters = args.cnn_hidden_channels // 2

        # Real stream
        real_layers = [GatedConvLayer(args.cnn_in_channels, num_real_filters)]
        for i in range(stream_length):
            dilation = 2 ** (i + 1)
            # dilation = 1
            real_layer = GatedConvLayer(num_real_filters, num_real_filters,
                                        local_condition=args.iso, dilation=dilation,
                                        preserve_size=True)
            real_layers.append(real_layer)

        # Complex stream
        complex_layers = [ComplexGatedConvLayer(args.cnn_in_channels, num_complex_filters,
                                                local_condition=args.iso)]
        for i in range(stream_length):
            dilation = 2 ** (i + 1)  # double dilation factor each layer
            complex_layer = ComplexGatedConvLayer(num_complex_filters, num_complex_filters,
                                                  local_condition=args.iso, dilation=dilation)
            complex_layers.append(complex_layer)

        # Combine outputs of complex and real streams
        self.pooling_conv = GatedConvLayer(num_real_filters + num_complex_filters, num_real_filters,
                                           local_condition=args.iso)
        self.output_conv = GatedConvLayer(num_real_filters, args.cnn_in_channels,
                                          normalize=False, layer_activation=None,
                                          local_condition=args.iso)

        self.real_stream = nn.ModuleList(real_layers)
        self.complex_stream = nn.ModuleList(complex_layers)

        self.residual = not args.interpolate

    def forward(self, x, c=None, class_labels=None):

        # Real stream
        real_out = x
        for r_layer in self.real_stream:
            real_out = r_layer(real_out, c)
            print(real_out.shape)
        # Complex stream
        complex_out = torch.rfft(x, signal_ndim=2)
        for c_layer in self.complex_stream:
            complex_out = c_layer(complex_out, c)
        complex_out = torch.irfft(complex_out, signal_ndim=2, signal_sizes=x.shape[2:])

        # Concatenate outputs of real and complex streams
        cat_streams = torch.cat([complex_out, real_out], dim=1)
        # Convolve over combined output
        out = self.pooling_conv(cat_streams, c)
        out = self.output_conv(out, c)

        print(out.shape)
        if self.residual:   # learn noise residual
            out = out + x

        return out


import argparse


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--run_on_test', nargs='+',
                        help='Evaluate a model (checkpoint) on test set. '
                        'Args: Checkpoint path, Test data path[, Save path]')

    parser.add_argument('-dd', '--data_dir', help='location of transformed data')
    parser.add_argument('-ts', '--test_split', help='Fraction of data to be used for validation',
                        default=0.2, type=float)
    parser.add_argument('-ds', '--data_subset', help='Fraction of crops per image to be used',
                        default=1.0, type=float)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--manual_seed', type=int,
                        help='manual seed, if not given resorts to random seed.')

    parser.add_argument('-sd', '--save_dir', type=str, metavar='PATH', default='',
                        help='path to save results and checkpoints to '
                             '(default: ../results/<model>/<current timestamp>)')
    parser.add_argument('--num-samples-to-log', type=int, metavar='N', default=32,
                        help='number of image samples to write to tensorboard each epoch'
                             ' (default: 32)')

    # training parameters
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-trb', '--train_batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size for training data (default: 256)')
    parser.add_argument('-teb', '--test_batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size for test data (default: 256)')

    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate (default: 0.005)')

    # model parameters
    parser.add_argument('--loss', type=str, default='MSELoss')
    parser.add_argument('--model', type=str, default='SimpleCNN')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--args_to_loss', action='store_true', default=False,
                        help='whether to pass the commandline arguments to the loss function')

    parser.add_argument('--resume', metavar='PATH', help='load from a path to a saved checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set (default: false)')

    # gpu/cpu
    parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU',
                        help='choose GPU to run on.')

    # CNN
    parser.add_argument('--cnn_in_channels', type=int, default=3)
    parser.add_argument('--cnn_hidden_channels', type=int, default=32)
    parser.add_argument('--cnn_num_hidden_layers', type=int, default=7)
    parser.add_argument('--interpolate', action='store_true', default=False,
                        help='interpolate rather than learn noise as an image residual')
    parser.add_argument('-ni', '--no_iso', action='store_true', default=False,
                        help='not to use image ISO values as extra conditioning data')

    # VGG loss
    parser.add_argument('--vgg_feature_layer', type=int, default=11,
                        help='VGG19 layer number from which to extract features')

    args = parser.parse_args(raw_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.iso = not args.no_iso

    if args.manual_seed is None:
        args.manual_seed = 42

    return args


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    conv = GatedConvLayer(3, 32, dilation=2, conv_residual=True, preserve_size=True)
    print(conv(x).shape)
    # model = HybridGatedCNN(parse_arguments())
    # model(x)
