import os

"""
# Instructions for users setting enviornment settings

# this must be done BEFORE importing and mga module. It only
# needs to be done once.

import os

# Turn JIT on/off (or set to "0" for pure Python)
# Default is off
os.environ["MGA_JIT_ENABLED"] = "1"

# Toggle 32/64 bit math
# Default is 64
os.environ["MGA_USE_32BIT"] = "1"

# Now import the library - this can be done downstream
from mga.mhmga import MGAProblem
"""

# Default to False for library usage; override via os.environ before import
_env_jit = str(os.environ.get("MGA_JIT_ENABLED", "0")).lower()
JIT_ENABLED = _env_jit in ("1", "true", "t", "yes", "y", "1.0", "on")

# Default to 64 bit for library usage; override via os.environ before import
_env_32bit = os.environ.get("MGA_USE_32BIT", "0").lower()
USE_32BIT = _env_32bit in ("1", "true", "t", "yes", "y", "1.0", "on")
