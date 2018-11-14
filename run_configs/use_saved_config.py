"""
Load a saved configuration and then run it

Assumes that the first commandline argument is the path to a saved config
"""
import sys
from pathlib import Path
import torch
from run_common import main


def load_and_run(arg):
    """Load config from given path and run"""
    path = str(Path(arg).resolve())
    args = torch.load(path)
    main(args)


if __name__ == "__main__":
    load_and_run(sys.argv[1])
