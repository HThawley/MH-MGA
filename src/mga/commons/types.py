import numpy as np
import warnings
import inspect

class defaults:
    def __init__(self, precision):
        if not isinstance(precision, int):
            raise TypeError("'precision' must be an int (32 or 64)")
        if precision == 32:
            self.float = np.float32
            self.int = np.int32
        elif precision == 64:
            self.float = np.float64
            self.int = np.int64
        else: 
            raise ValueError("'precision' must be 32 or 64")
        self.precision = precision
        self.settable = True

    def update_precision(self, precision):
        if not self.settable: 
            warnings.warn("Set precision must be called *before* importing other mga modules "\
                    "to ensure precision is set correctly.", UserWarning)

        if not isinstance(precision, int):
            raise TypeError("'precision' must be an int (32 or 64)")
        if precision == 32:
            self.float = np.float32
            self.int = np.int32
        elif precision == 64:
            self.float = np.float64
            self.int = np.int64
        else: 
            raise ValueError("'precision' must be 32 or 64")
        self.precision = precision

    def __iter__(self):
        self.settable = False
        return iter((self.int, self.float))
    
    def __repr__(self):
        return f"Default data types int and float of precision {self.precision}"

DEFAULTS = defaults(32)

