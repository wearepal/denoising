"""Load configuration from a JSON file and then run it"""
import sys
import json
from pathlib import Path
from run_common import run


def main(path_arg):
    """Load JSON file and run"""
    with Path(path_arg).resolve().open() as json_file:
        args = json.load(json_file)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1])
