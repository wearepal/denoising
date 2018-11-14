import sys
from pathlib import Path

# we need to import main.py from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import main, parse_arguments


def run(args: dict):
    """Run the pytorch model with the configuration given in args"""
    # convert the dictionary into commandline args
    arg_list = []
    for k, v in args.items():
        arg_list += [f"--{k}", str(v)]
    main(parse_arguments(arg_list))
