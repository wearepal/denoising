"""Configuration loading and parsing"""
from dataclasses import asdict, dataclass
import random
from pathlib import Path
from typing import Optional

import yaml
import dacite

import torch


@dataclass
class Settings:
    # general
    test_data_dir: Optional[Path]
    results_dir: Optional[Path]
    data_dir: Optional[Path]
    test_split: float
    data_subset: float
    workers: int
    cuda: bool
    random_seed: bool
    save_dir: Optional[Path]
    num_samples_to_log: int
    resume: Optional[Path]
    evaluate: bool

    # training parameters
    epochs: int
    start_epoch: int
    pretrain_epochs: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float

    # optimization parameters
    beta1: float
    beta2: float
    gen_learning_rate: float
    disc_learning_rate: float
    disc_iters: int

    # loss parameters
    loss: str
    content_loss: str
    adv_loss: str
    adv_weight: float
    args_to_loss: bool

    # model parameters
    model: str
    generator: str
    discriminator: str
    optim: str

    # gpu/cpu
    gpu_num: int
    multi_gpu: bool

    # CNN
    cnn_in_channels: int
    cnn_hidden_channels: int
    cnn_hidden_layers: int
    residual: bool
    iso: bool
    use_class: bool
    learn_beta: bool

    # VGG loss
    vgg_feature_layer: int

    # misc
    num_classes: int = -1
    seed: int = -1

    def asdict(self) -> dict:
        return asdict(self)


def parse_arguments(config_file: str) -> Settings:
    """This function basically just checks if all config values have the right type."""
    config_path = Path(config_file)
    with config_path.open("r") as fp:
        args_dict = yaml.safe_load(fp)
    args = dacite.from_dict(Settings, args_dict, config=dacite.Config(cast=[Path], strict=True))
    args.num_classes = 3 if args.use_class else 0
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.random_seed:
        args.seed = random.randint(1, 100000)
    else:
        args.seed = 42
    return args
