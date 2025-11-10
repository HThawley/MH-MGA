import numpy as np
import sys
import importlib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", type=str, required=True, help="module imported for test")
args = parser.parse_args()


def test_precision_settability(module: str):
    try:
        from mga.commons.types import DEFAULTS

        assert DEFAULTS.settable, "'settable' not initialized as `True`"
        module = importlib.import_module(module)
        if hasattr(module, "INT") or hasattr(module, "FLOAT"):
            assert not DEFAULTS.settable, "'settable' not updated to `False` on import"
        else:
            assert (
                DEFAULTS.settable
            ), "'settable' updated to 'False` when 'INT' and 'FLOAT' are not resolved by importer"
    except AssertionError as e:
        print(f"Assertion Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Failed to import module: {module}")
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    test_precision_settability(args.m)
