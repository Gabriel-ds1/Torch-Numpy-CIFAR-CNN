"""
cifar10_data_loader.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Utility module for downloading, extracting, loading, splitting, and augmenting the CIFAR-10 dataset
             using a NumPy-compatible backend (CPU or GPU). Includes functions for batch loading, train/val/test
             splitting, optional normalization, and a variety of single-image augmentation techniques.
Published: 04-26-2025
"""

import os
import urllib.request
import tarfile
import pickle
from numpy_resnet.utils import backend

# Remote URL and local file paths for CIFAR-10
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TAR = "cifar-10-python.tar.gz"
CIFAR_DIR = "cifar-10-batches-py"

def download_and_extract_cifar10():
    """
    Download the CIFAR-10 tarball if not present, and extract its contents into CIFAR_DIR.

    Checks for existing extracted directory first; if missing, downloads the .tar.gz and unpacks it.
    """
    # Only extract if directory does not exist
    if not os.path.exists(CIFAR_DIR):
        # Download archive if not already downloaded
        if not os.path.exists(CIFAR_TAR):
            print("Downloading CIFAR-10 dataset...")
            urllib.request.urlretrieve(CIFAR_URL, CIFAR_TAR)
        print("Extracting dataset...")
        with tarfile.open(CIFAR_TAR) as tar:
            tar.extractall()

def load_batch(batch_path):
    """
    Load a single CIFAR-10 batch from disk and return data and labels arrays.

    Args:
        batch_path (str): Path to a CIFAR-10 batch file (e.g., data_batch_1 or test_batch).

    Returns:
        data (backend.np.ndarray): Array of shape (N, 3, 32, 32), normalized to [0,1].
        labels (backend.np.ndarray): Integer class labels of shape (N,).
    """
    # Read raw pickle data (bytes keys)
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # raw_data is a numpy array; turn it to backend.np immediately (important if using CuPy)
    data = backend.np.asarray(batch[b'data'], dtype=backend.np.float32)
    # now # Reshape from (N, 3072) to (N, 3, 32, 32) and normalize pixel values
    data = data.reshape(-1, 3, 32, 32) / backend.np.float32(255.0)

    # Load labels as integer array
    labels = backend.np.asarray(batch[b'labels'], dtype=backend.np.int64)
    return data, labels

def load_cifar10(val_split=0.1, augment=False, normalize=False):
    """
    Download (if needed), load, and split CIFAR-10 into train, validation, and test sets.

    Args:
        val_split (float): Fraction of training data to reserve for validation.
        augment (bool): Whether to apply random augmentations to training images.
        normalize (bool): Whether to apply channel-wise mean/std normalization.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test): Tuples of arrays.
    """
    # Ensure data is available on local disk
    download_and_extract_cifar10()

    # Load all 5 training batches
    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(CIFAR_DIR, f'data_batch_{i}'))
        x_train.append(data)
        y_train.append(labels)
    # Concatenate into full train set
    x_train = backend.np.concatenate(x_train)
    y_train = backend.np.concatenate(y_train)

    # Shuffle training data before splitting
    indices = backend.np.arange(len(x_train))
    backend.np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Split off validation portion
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Load test batch (unshuffled)
    x_test, y_test = load_batch(os.path.join(CIFAR_DIR, 'test_batch'))

    # Apply optional augmentations to train set only
    if augment:
        x_train = augment_batch(x_train)
    
    # Apply optional normalization to all splits
    if normalize:
        mean = backend.np.array([0.4914,0.4822,0.4465]).reshape(1,3,1,1)
        std  = backend.np.array([0.2023,0.1994,0.2010]).reshape(1,3,1,1)
        x_train = (x_train - mean) / std
        x_val   = (x_val   - mean) / std
        x_test  = (x_test  - mean) / std

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# AUGMENTATION METHODS (single-image versions)
def augment_single(image):
    """
    Apply a series of random augmentations to a single CIFAR-10 image.

    Augmentations include padding+random crop, horizontal flip, brightness,
    contrast, saturation changes, and small rotations.

    Args:
        image (backend.np.ndarray): Array of shape (3, 32, 32) with values in [0,1].

    Returns:
        augmented (backend.np.ndarray): Augmented image clipped to [0,1].
    """
    # Reflect-pad to 40×40, then random crop back to 32×32
    padded = backend.np.pad(image, ((0, 0), (4, 4), (4, 4)), mode='reflect')
    top = backend.np.random.randint(0, 8)
    left = backend.np.random.randint(0, 8)
    image = padded[:, top:top+32, left:left+32]

    # 50% chance horizontal flip
    if backend.np.random.rand() < 0.5:
        image = image[:, :, ::-1]

    # Random brightness adjustment
    image += backend.np.random.uniform(-0.2, 0.2)

    # Random contrast adjustment around per-channel mean
    mean = image.mean(axis=(1, 2), keepdims=True)
    image = (image - mean) * backend.np.random.uniform(0.8, 1.2) + mean

    # Random saturation blend with grayscale version
    gray = image.mean(axis=0, keepdims=True)
    sat_factor = backend.np.random.uniform(0.8, 1.2)
    image = image * sat_factor + gray * (1 - sat_factor)

    # Random rotation within about 15 degrees (Use cupyx scipy if running on GPU)
    if backend.IS_GPU:
        try:
            from cupyx.scipy.ndimage import rotate
        except ImportError:
            # fallback if cupyx is missing
            from scipy.ndimage import rotate
    else:
        from scipy.ndimage import rotate
    angle = backend.np.random.uniform(-15, 15)
    for c in range(3):
        image[c] = rotate(image[c], angle, reshape=False, mode='reflect')
    # Clip to valid range and return
    return backend.np.clip(image, 0.0, 1.0)

def augment_batch(images):
    """
    Apply augment_single to each image in a batch.

    Args:
        images (backend.np.ndarray): Array of shape (N, 3, 32, 32).

    Returns:
        augmented_batch (backend.np.ndarray): Augmented batch of same shape.
    """
    return backend.np.stack([augment_single(img) for img in images])
