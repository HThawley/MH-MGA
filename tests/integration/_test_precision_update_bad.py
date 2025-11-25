import numpy as np
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", type=int, required=True, help="precision for test")
args = parser.parse_args()


def test_precision_update_bad(precision):
    from mga.commons.types import DEFAULTS

    try:
        DEFAULTS.update_precision(precision)
    except Exception as e:
        if not isinstance(precision, int):
            if not isinstance(e, TypeError):
                print(
                    f"Raised wrong error with input of type: ({type(precision)}). Expected TypeError, got: {type(e)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                sys.exit(0)
        if not precision in (32, 64):
            if not isinstance(e, ValueError):
                print(
                    f"Raised wrong error with input of value: ({precision}). Expected ValueError, got: {type(e)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                sys.exit(0)
        raise e


if __name__ == "__main__":
    test_precision_update_bad(args.p)
