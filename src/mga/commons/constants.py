import os
import numpy as np

"""
# Instructions for users turning jit on/off
import os

# Turn JIT ON (or set to "0" for pure Python)
os.environ["MGA_JIT_ENABLED"] = "1"

# Now import the library
from mga.mhmga import MGAProblem
"""

# Default to False for library usage; override via os.environ before import
_env_jit = str(os.environ.get("MGA_JIT_ENABLED", "0")).lower()
JIT_ENABLED = _env_jit in ("1", "true", "t", "yes", "y", "1.0", "on")

INT = np.int64
FLOAT = np.float64
