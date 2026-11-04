"""Educational insurance claim prediction utilities."""

import platform
import sys
import warnings

__version__ = "0.1.0"

# On arm64 macOS, numpy linked to Apple Accelerate raises these warnings on clean,
# finite data, and results match a non-BLAS calculation. See "Known issues" in the README.
if sys.platform == "darwin" and platform.machine() == "arm64":
    warnings.filterwarnings(
        "ignore",
        message=r"(divide by zero|overflow|invalid value) encountered in matmul",
        category=RuntimeWarning,
    )
