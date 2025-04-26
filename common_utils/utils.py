"""
run_dir_utils.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Utility for creating sequential run directories for model checkpoints and logs.
             Ensures a new, uniquely numbered folder is created for each training run.
Published: 04-26-2025
"""

import os

def get_new_run_dir(model_type="pytorch", base_dir="model_ckpt/"):
    """
    Create a new run directory for storing checkpoints or logs, with automatic numbering.

    Each run directory is named with a three-digit sequence, incremented from existing folders.

    Args:
        model_type (str): Subdirectory under base_dir to categorize runs (e.g., 'pytorch' or 'numpy').
        base_dir (str): Base directory where run subfolders reside.

    Returns:
        str: Path to the newly created run directory (e.g., 'model_ckpt/pytorch/002').

    Raises:
        FileExistsError: If the new run directory already exists (should not occur under normal usage).
    """
    # Ensure model_type folder exists
    os.makedirs(os.path.join(base_dir, model_type), exist_ok=True)

    # List existing run directories under base_dir that are purely numeric
    run_base_dir = os.path.join(base_dir, model_type)
    existing = [d for d in os.listdir(run_base_dir) if d.isdigit()]

    if existing:
        # Determine the highest existing run number and increment
        max_run = max(int(d) for d in existing)
        new_run = f"{max_run+1:03d}"  # 3-digit format
    else:
        # First run defaults to '001'
        new_run = "001"
        
    # Create the new run directory path
    new_dir = os.path.join(run_base_dir, new_run)
    os.makedirs(new_dir, exist_ok=False)
    return new_dir