#!/usr/bin/env python3
"""Run Terminal Guidance."""

# Set OpenMP threads FIRST, before any imports load NCNN/NumPy
import os
os.environ["OMP_NUM_THREADS"] = "4"

from src.app import main

if __name__ == "__main__":
    main()
