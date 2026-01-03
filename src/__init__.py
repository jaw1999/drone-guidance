"""Terminal Guidance - Drone Companion Computer for Target Tracking"""

# Set thread counts before importing NumPy/NCNN (2 threads optimal on Pi 5)
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

__version__ = "0.1.0"
