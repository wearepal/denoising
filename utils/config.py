"""Configuration loading and parsing"""
import configparser
from contextlib import contextmanager

import torch


class SectionSetter:
    """Object that is used to set values from a configparser section to a settings object"""
    def __init__(self, settings_obj, section_obj):
        """Constructor

        Args:
            settings_obj: instance of Settings
            section_obj: a section from a configparser
        """
        self.settings = settings_obj
        self.section = section_obj

    def set_str(self, key):
        """Set string configuration value from `section`"""
        self.__set(self.section.get, key, "string")

    def set_int(self, key):
        """Set integer configuration value from `section`"""
        self.__set(self.section.getint, key, "integer")

    def set_float(self, key):
        """Set float configuration value from `section`"""
        self.__set(self.section.getfloat, key, "float")

    def set_bool(self, key):
        """Set boolean configuration value from `section`"""
        self.__set(self.section.getboolean, key, "boolean")

    def __set(self, getter, key, type_name):
        """Set a value from the getter where the getter is a function"""
        try:
            value = getter(key)
        except ValueError:
            raise ValueError(f"Config file: the value for \"{key}\" has to be of type {type_name}.")
        self.settings.set(key, value)


class Settings:
    """Object that holds the configuration"""
    def __init__(self, config_file):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_file)
        self._key_list = []

    def __find_section(self, section_name):
        """Find section"""
        for section in self.parser.sections():
            if section.lower() == section_name.lower():
                return self.parser[section]
        self.parser.add_section(section_name)
        return self.parser[section_name]

    def get_section(self, section_name):
        """Return an object that represents the section with the given name

        Use this function if you don't want to use the context manager.
        """
        section = self.__find_section(section_name)
        return SectionSetter(self, section)

    @contextmanager
    def section(self, section_name):
        """Context manager for adding values from a section"""
        section_setter = self.get_section(section_name)
        yield section_setter

    def set(self, key, value):
        """Set the given value for the given key

        You should always use this method to set a value instead of using __setattr__ directly,
        because otherwise the dictionary representation returned by state_dict won't contain it.
        """
        self._key_list.append(key)
        self.__setattr__(key, value)

    def state_dict(self):
        """Convert this settings object to a dictionary"""
        return {key: getattr(self, key) for key in self._key_list}

    def load_state_dict(self, dictionary):
        """Load settings from a dictionary"""
        for key, value in dictionary.iter():
            self.set(key, value)


def parse_arguments(config_file):
    """
    This function basically just checks if all config values have the right type.
    """
    args = Settings(config_file)

    with args.section('general') as s:
        s.set_str('test_data_dir')
        s.set_str('results_dir')
        s.set_str('data_dir')
        s.set_float('test_split')
        s.set_float('data_subset')
        s.set_int('workers')
        s.set_bool('cuda')
        s.set_bool('random_seed')
        s.set_str('save_dir')
        s.set_str('num_samples_to_log')
        s.set_str('resume')
        s.set_bool('evaluate')

    # training parameters
    with args.section('training') as s:
        s.set_int('epochs')
        s.set_int('start_epoch')
        s.set_int('pretrain_epochs')
        s.set_int('train_batch_size')
        s.set_int('test_batch_size')
        s.set_float('learning_rate')

    # optimization parameters
    with args.section('optimization') as s:
        s.set_float('beta1')
        s.set_float('beta2')
        s.set_float('gen_learning_rate')
        s.set_float('disc_learning_rate')
        s.set_int('disc_iters')

    # loss parameters
    with args.section('loss') as s:
        s.set_str('loss')
        s.set_str('content_loss')
        s.set_str('adv_loss')
        s.set_float('adv_weight')
        s.set_bool('args_to_loss')

    # model parameters
    with args.section('model') as s:
        s.set_str('model')
        s.set_str('generator')
        s.set_str('discriminator')
        s.set_str('optim')

    # gpu/cpu
    with args.section('gpu') as s:
        s.set_int('gpu_num')
        s.set_bool('multi_gpu')

    # CNN
    with args.section('cnn') as s:
        s.set_int('cnn_in_channels')
        s.set_int('cnn_hidden_channels')
        s.set_int('cnn_hidden_layers')
        s.set_bool('residual')
        s.set_bool('iso')
        s.set_bool('use_class')
        s.set_bool('learn_beta')

    # VGG loss
    args.get_section('vgg').set_str('vgg_feature_layer')

    args.set('num_classes', 3 if args.use_class else 0)
    args.set('cuda', args.cuda and torch.cuda.is_available())

    if args.random_seed:
        args.set('seed', random.randint(1, 100000))
    else:
        args.set('seed', 42)

    return args
