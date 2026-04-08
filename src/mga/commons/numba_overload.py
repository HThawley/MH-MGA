"""
Numba functions are overloaded to allow for JIT to be switched off,
allowing for debugging with the Python interpreter instead.
"""
import numpy as np
from numpy.typing import NDArray

from mga.commons.constants import JIT_ENABLED

if JIT_ENABLED:  # noqa: C901
    from numba import njit, prange, int32, int64, float32, float64, boolean
    from numba.experimental import jitclass

else:
    def jitclass(spec):
        def decorator(cls):
            return cls

        return decorator

    def njit(func=None, **kwargs):
        if func is not None:
            return func

        def wrapper(f):
            return f

        return wrapper

    def prange(*args):
        return range(*args)

    class _Int64:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.int64]

    class _Int32:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.int32]

    class _Float64:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.float64]

    class _Float32:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.float32]

    class _Boolean:
        @classmethod
        def __class_getitem__(cls, key):
            return NDArray[np.bool_]

    int64 = _Int64
    int32 = _Int32
    float64 = _Float64
    float32 = _Float32
    boolean = _Boolean
