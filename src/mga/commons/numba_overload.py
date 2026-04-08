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

    def _make_mock_type(np_type):
        class MockType:
            def __new__(cls, value=0):
                return np_type(value)

            @classmethod
            def __class_get_item__(cls, key):
                return NDArray[np_type]

        MockType.__name__ = f"_{np_type.__name__.capitalize()}"
        return MockType

    int64 = _make_mock_type(np.int64)
    int32 = _make_mock_type(np.int32)
    float64 = _make_mock_type(np.float64)
    float32 = _make_mock_type(np.float32)
    boolean = _make_mock_type(np.bool_)
