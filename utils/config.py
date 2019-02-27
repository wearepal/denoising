"""Configuration loading and parsing"""
import configparser

import torch


class Settings:
    """Object that holds the configuration"""
    def __init__(self, config_file):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_file)
        self._key_list = []

    def set_str(self, section, key):
        """Set string configuration value from `section`"""
        self.__set(self.__section(section).get, key, "string")

    def set_int(self, section, key):
        """Set integer configuration value from `section`"""
        self.__set(self.__section(section).getint, key, "int")

    def set_float(self, section, key):
        """Set float configuration value from `section`"""
        self.__set(self.__section(section).getfloat, key, "float")

    def set_bool(self, section, key):
        """Set boolean configuration value from `section`"""
        self.__set(self.__section(section).getboolean, key, "boolean")

    def __section(self, section_name):
        """Find section"""
        for section in self.parser.sections():
            if section.lower() == section_name.lower():
                return self.parser[section]
        self.parser.add_section(section_name)
        return self.parser[section_name]

    def __set(self, getter, key, type_name):
        """Set a value from the getter where the getter is a function"""
        try:
            value = getter(key)
        except ValueError:
            raise ValueError(f"Config file: the value for \"{key}\" has to be of type {type_name}.")
        self._key_list.append(key)
        self.__setattr__(key, value)

    def state_dict(self):
        """Convert this settings object to a dictionary"""
        return {key: getattr(self, key) for key in self._key_list}

    def load_state_dict(self, dictionary):
        """Load settings from a dictionary"""
        for key, value in dictionary.iter():
            self.__setattr__(key, value)


def parse_arguments(config_file):
    """
    This function basically just checks if all config values have the right type.
    """
    args = Settings(config_file)

    args.set_str('general', 'test_data_dir')
    args.set_str('general', 'results_dir')
    args.set_str('general', 'data_dir')
    args.set_float('general', 'test_split')
    args.set_float('general', 'data_subset')
    args.set_int('general', 'workers')
    args.set_bool('general', 'cuda')
    args.set_bool('general', 'random_seed')
    args.set_str('general', 'save_dir')
    args.set_str('general', 'num_samples_to_log')
    args.set_str('general', 'resume')
    args.set_bool('general', 'evaluate')

    # training parameters
    args.set_int('training', 'epochs')
    args.set_int('training', 'start_epoch')
    args.set_int('training', 'pretrain_epochs')
    args.set_int('training', 'train_batch_size')
    args.set_int('training', 'test_batch_size')
    args.set_float('training', 'learning_rate')

    # optimization parameters
    args.set_float('optimization', 'beta1')
    args.set_float('optimization', 'beta2')
    args.set_float('optimization', 'gen_learning_rate')
    args.set_float('optimization', 'disc_learning_rate')
    args.set_int('optimization', 'disc_iters')

    # loss parameters
    args.set_str('loss', 'loss')
    args.set_str('loss', 'content_loss')
    args.set_str('loss', 'adv_loss')
    args.set_float('loss', 'adv_weight')
    args.set_bool('loss', 'args_to_loss')

    # model parameters
    args.set_str('model', 'model')
    args.set_str('model', 'generator')
    args.set_str('model', 'discriminator')
    args.set_str('model', 'optim')

    # gpu/cpu
    args.set_int('gpu', 'gpu_num')
    args.set_bool('gpu', 'multi_gpu')

    # CNN
    args.set_int('cnn', 'cnn_in_channels')
    args.set_int('cnn', 'cnn_hidden_channels')
    args.set_int('cnn', 'cnn_hidden_layers')
    args.set_bool('cnn', 'residual')
    args.set_bool('cnn', 'iso')
    args.set_bool('cnn', 'use_class')
    args.set_bool('cnn', 'learn_beta')

    # VGG loss
    args.set_str('vgg', 'vgg_feature_layer')

    args.num_classes = 3 if args.use_class else 0
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.random_seed:
        args.seed = random.randint(1, 100000)
    else:
        args.seed = 42

    return args
