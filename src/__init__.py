"""Terminal Guidance - Drone Companion Computer for Target Tracking"""

# Set OpenMP threads BEFORE any imports that might use them (NCNN, NumPy, etc.)
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

__version__ = "0.1.0"
