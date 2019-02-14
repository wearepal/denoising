"""Configuration loading and parsing"""
import configparser
import torch


class Settings:
    """Object that holds the configuration"""
    def __init__(self):
        self._key_list = []

    def set_str(self, section, name):
        """Set string configuration value from `section`"""
        self.__set(name, section.get(name))

    def set_int(self, section, name):
        """Set integer configuration value from `section`"""
        try:
            self.__set(name, section.getint(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type int.")

    def set_float(self, section, name):
        """Set float configuration value from `section`"""
        try:
            self.__set(name, section.getfloat(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type float.")

    def set_bool(self, section, name):
        """Set boolean configuration value from `section`"""
        try:
            self.__set(name, section.getboolean(name))
        except ValueError:
            raise ValueError(f"Config file: the value for \"{name}\" has to be of type boolean.")

    def __set(self, name, value):
        self._key_list.append(name)
        self.__setattr__(name, value)
        if value is None:
            print(f"Warning: no value specified for \"{name}\" (value is therefore set to None).")

    def state_dict(self):
        """Convert this settings object to a dictionary"""
        return {name: getattr(self, name) for name in self._key_list}

    def load_state_dict(self, dictionary):
        """Load settings from a dictionary"""
        for key, value in dictionary.iter():
            self.__setattr__(key, value)


def parse_arguments(config_file):
    """
    This function basically just checks if all config values have the right type.
    """
    args = Settings()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    general_config = get_section(parser, 'general')
    args.set_str(general_config, 'test_data_dir')
    args.set_str(general_config, 'results_dir')
    args.set_str(general_config, 'data_dir')
    args.set_float(general_config, 'test_split')
    args.set_float(general_config, 'data_subset')
    args.set_int(general_config, 'workers')
    args.set_bool(general_config, 'cuda')
    args.set_bool(general_config, 'random_seed')
    args.set_str(general_config, 'save_dir')
    args.set_str(general_config, 'num_samples_to_log')
    args.set_str(general_config, 'resume')
    args.set_bool(general_config, 'evaluate')

    # training parameters
    training_config = get_section(parser, 'training')
    args.set_int(training_config, 'epochs')
    args.set_int(training_config, 'start_epoch')
    args.set_int(training_config, 'pretrain_epochs')
    args.set_int(training_config, 'train_batch_size')
    args.set_int(training_config, 'test_batch_size')
    args.set_float(training_config, 'learning_rate')

    # optimization parameters
    optimization_config = get_section(parser, 'optimization')
    args.set_float(optimization_config, 'beta1')
    args.set_float(optimization_config, 'beta2')
    args.set_float(optimization_config, 'gen_learning_rate')
    args.set_float(optimization_config, 'disc_learning_rate')
    args.set_int(optimization_config, 'disc_iters')

    # loss parameters
    loss_config = get_section(parser, 'loss')
    args.set_str(loss_config, 'loss')
    args.set_str(loss_config, 'content_loss')
    args.set_str(loss_config, 'adv_loss')
    args.set_float(loss_config, 'adv_weight')
    args.set_bool(loss_config, 'args_to_loss')

    # model parameters
    model_config = get_section(parser, 'model')
    args.set_str(model_config, 'model')
    args.set_str(model_config, 'generator')
    args.set_str(model_config, 'discriminator')
    args.set_str(model_config, 'optim')

    # gpu/cpu
    gpu_config = get_section(parser, 'GPU')
    args.set_int(gpu_config, 'gpu_num')

    # CNN
    cnn_config = get_section(parser, 'CNN')
    args.set_int(cnn_config, 'cnn_in_channels')
    args.set_int(cnn_config, 'cnn_hidden_channels')
    args.set_int(cnn_config, 'cnn_hidden_layers')
    args.set_bool(cnn_config, 'residual')
    args.set_bool(cnn_config, 'iso')
    args.set_bool(cnn_config, 'use_class')

    # VGG loss
    vgg_config = get_section(parser, 'VGG')
    args.set_str(vgg_config, 'vgg_feature_layer')

    args.num_classes = 3 if args.use_class else 0
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.random_seed:
        args.seed = random.randint(1, 100000)
    else:
        args.seed = 42

    return args


def get_section(parser, name):
    """Get section of the config parser. Creates section if it doesn't exist"""
    if not parser.has_section(name):
        parser.add_section(name)
    return parser[name]
