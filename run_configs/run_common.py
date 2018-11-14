import sys
from pathlib import Path

# we need to import main.py from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import main, parse_arguments


def run(args: dict):
    """Run the pytorch model with the configuration given in args"""
    # convert the dictionary into commandline args
    arg_list = []
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if not value:
                arg_list += [f"--no_{key}"]
            continue
        arg_list += [f"--{key}", str(value)]
    main(parse_arguments(arg_list))
