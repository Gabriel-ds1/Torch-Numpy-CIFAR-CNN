"""
backend_utils.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Backend configuration and GPU monitoring utilities for NumPy/CuPy-based computations.
             Includes dynamic backend switching, GPU synchronization, memory profiling, and timing context manager.
Published: 04-26-2025
"""

import time
from contextlib import contextmanager
import numpy as _np
# Attempt to import CuPy for GPU acceleration; fall back to None if unavailable
try:
    import cupy as _cp
except ImportError:
    _cp = None

# Global flags and aliases for active backend
IS_GPU = False
np = _np # Default to NumPy
cp = _cp # CuPy if available, else None


def set_backend(device):
    """
    Select compute backend: 'gpu' for CuPy (if available), otherwise NumPy.

    Modifies global 'np' alias and 'IS_GPU' flag for downstream operations.

    Args:
        device (str): 'gpu' to attempt CuPy, anything else for NumPy.
    """
    global np, IS_GPU
    if device == "gpu" and cp is not None:
        try:
            # quick test allocation on GPU
            cp.zeros((1,)).sum()
            np = cp
            IS_GPU = True
            print("Running on GPU with CuPy")
        except Exception as e:
            # Fallback to NumPy if CuPy fails
            np = _np
            IS_GPU = False
            print(f"CuPy test failed ({e}), falling back to NumPy")
    else:
        # Force CPU backend
        np = _np
        IS_GPU = False
        print("Running on CPU with NumPy")

def sync_gpu():
    """
    Synchronize GPU operations to ensure all kernels complete.

    Only effective if using CuPy backend; otherwise does nothing.
    """
    if IS_GPU:
        cp.cuda.Device(0).synchronize()

def monitor_gpu():
    """
    Print current CuPy memory pool usage and total device memory.

    Useful for debugging memory leaks and profiling GPU usage.
    """
    if not IS_GPU:
        print("Not running on GPU")
        return
    # Memory pool tracking
    mempool = cp.get_default_memory_pool()
    used = mempool.used_bytes()
    total = cp.cuda.Device(0).mem_info[1]
    used_mb = used / (1024 ** 2)
    total_mb = total / (1024 ** 2)

    print(f"[GPU] Used by CuPy memory pool: {used_mb:.2f} MB / {total_mb:.2f} MB total")

class GPUMemoryProfiler:
    """
    Epoch-wise GPU memory profiler for clearing and logging CuPy memory pool.

    Attributes:
        clear_every (int): Frequency (in epochs) to free all memory pool blocks.
        log_every (int): Frequency to log memory usage.
    """
    def __init__(self, clear_every: int = 10, log_every: int = 1):
        self.clear_every = clear_every
        self.log_every = log_every

    def log(self, epoch: int, logger=None):
        """
        Log memory pool usage at the given epoch.

        Args:
            epoch (int): Current epoch index (0-based).
            logger: Optional logger to write info; prints if None.
        """
        if not IS_GPU:
            return
        if epoch % self.log_every == 0:
            used = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
            total = cp.cuda.Device(0).mem_info[1] / (1024 ** 2)
            msg = f"[GPU] Epoch {epoch+1} | Memory pool: {used:.2f} MB / {total:.2f} MB"
            print(msg) if logger is None else logger.info(msg)

    def clear(self, epoch: int, logger=None):
        """
        Clear GPU memory pool at the given epoch to prevent fragmentation.

        Args:
            epoch (int): Current epoch index (0-based).
            logger: Optional logger to write info; prints if None.
        """
        if not IS_GPU:
            return
        if epoch % self.clear_every == 0:
            cp.get_default_memory_pool().free_all_blocks()
            msg = f"[GPU] Epoch {epoch+1} | Cleared memory pool"
            print(msg) if logger is None else logger.info(msg)

@contextmanager
def profile_block(name="Block", logger=None):
    """
    Context manager to time execution and track GPU memory delta for a code block.

    Args:
        name (str): Identifier for the profiling block.
        logger: Optional logger to record profiling info; prints to stdout if None.
    """
    # Start timing and optional GPU sync
    start_time = time.time()
    start_mem = 0

    if IS_GPU:
        cp.cuda.Device(0).synchronize()
        start_mem = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)

    yield  # Run the code block

    # End timing and memory measurement
    if IS_GPU:
        cp.cuda.Device(0).synchronize()
        end_mem = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
    else:
        end_mem = 0

    elapsed = time.time() - start_time
    msg = f"{name} | Time: {elapsed:.3f}s | GPU Memory: {start_mem:.2f} -> {end_mem:.2f} MB"
    if logger:
        logger.info(msg)
    else:
        print(msg)