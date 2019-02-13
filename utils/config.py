"""Configuration loading and parsing"""
import configparser


class Settings:
    """Object that holds the configuration"""
    def __init__(self):
        self._key_list = []

    def set_str(self, section, name):
        """Set string configuration value from `section`"""
        self.__set(name, section.get(name))

    def set_int(self, section, name):
        """Set integer configuration value from `section`"""
        self.__set(name, section.getint(name))

    def set_float(self, section, name):
        """Set float configuration value from `section`"""
        self.__set(name, section.getfloat(name))

    def set_bool(self, section, name):
        """Set boolean configuration value from `section`"""
        self.__set(name, section.getboolean(name))

    def __set(self, name, value):
        self._key_list.append(name)
        self.__setattr__(name, value)

    def to_dict(self):
        """Convert this settings object to a dictionary"""
        return {name: getattr(self, name) for name in self._key_list}

    def from_dict(self, dictionary):
        """Load settings from a dictionary"""
        for key, value in dictionary.iter():
            self.__setattr__(key, value)


def parse_arguments(config_file):
    args = Settings()
    parser = configparser.ConfigParser()
    parser.read(config_file)

    general_config = parser['general']
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

    # training parameters
    training_config = parser['training']
    args.set_int(training_config, 'epochs')
    args.set_int(training_config, 'start_epoch')
    args.set_int(training_config, 'train_batch_size')
    args.set_int(training_config, 'test_batch_size')
    args.set_float(training_config, 'learning_rate')

    # model parameters
    model_config = parser['model']
    args.set_str(model_config, 'loss')
    args.set_str(model_config, 'model')
    args.set_str(model_config, 'optim')
    args.set_bool(model_config, 'args_to_loss')
    args.set_str(model_config, 'resume')
    args.set_bool(model_config, 'evaluate')

    # gpu/cpu
    gpu_config = parser['GPU']
    args.set_int(gpu_config, 'gpu_num')

    # CNN
    cnn_config = parser['CNN']
    args.set_int(cnn_config, 'cnn_in_channels')
    args.set_int(cnn_config, 'cnn_hidden_channels')
    args.set_int(cnn_config, 'cnn_hidden_layers')
    args.set_bool(cnn_config, 'residual')
    args.set_bool(cnn_config, 'iso')
    args.set_bool(cnn_config, 'use_class')

    # VGG loss
    vgg_config = parser['VGG']
    args.set_int(vgg_config, 'vgg_feature_layer')

    args.num_classes = 3 if args.use_class else 0

    if args.random_seed:
        args.seed = random.randint(1, 100000)
    else:
        args.seed = 42

    return args
