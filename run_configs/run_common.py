import sys

# try different ways to import main.py; it's difficult because main.py is in the parent directory
sys.path.append('.')
sys.path.append('..')
from main import main, parse_arguments


def run(args: dict):
    """Run the pytorch model with the configuration given in args"""
    # convert the dictionary into commandline args
    arg_list = []
    for k, v in args.items():
        arg_list += [f"--{k}", str(v)]
    main(parse_arguments(arg_list))
