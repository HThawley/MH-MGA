import numpy as np

from mga.commons.constants import USE_32BIT
from mga.commons.numba_overload import int32, int64, float32, float64, boolean

if USE_32BIT:
    npint = np.int32
    npintp = np.int64  # native 64 bit indexing on modern machines
    npfloat = np.float32

    nbint = int32
    nbintp = int64  # native 64 bit indexing on modern machines
    nbfloat = float32
    boolean
else:
    npint = np.int64
    npintp = np.int64  # native 64 bit indexing on modern machines
    npfloat = np.float64

    nbint = int64
    nbintp = int64  # native 64 bit indexing on modern machines
    nbfloat = float64
    boolean
