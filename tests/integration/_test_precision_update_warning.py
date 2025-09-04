import numpy as np 
import sys
import importlib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-m', type=str, required=True, help='module imported for test')
args = parser.parse_args()

def test_precision_update_warning(module:str):
    module = importlib.import_module(module)
    from mga.commons.types import DEFAULTS
    DEFAULTS.update_precision(32)

if __name__ == "__main__":
    test_precision_update_warning(args.m)