"""
cifar10_torch_dataset.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: PyTorch Dataset implementation for CIFAR-10, supporting optional on-the-fly augmentation,
             normalization, and tensor conversion. Also provides an unnormalize utility for visualization.
Published: 04-26-2025
"""

import numpy as np
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import rotate
from torch.utils.data import Dataset
import torch

class CIFARDataset(Dataset):
    """
    PyTorch Dataset for CIFAR-10 images and labels.

    Supports optional data augmentation on each sample and applies standard normalization.
    Converts NumPy arrays to torch tensors for model training.
    """
    def __init__(self, images, labels, augment=False):
        """
        Initialize the CIFAR-10 dataset wrapper.

        Args:
            images (np.ndarray): Array of shape (N, 3, 32, 32) with pixel values in [0,1].
            labels (np.ndarray): Array of shape (N,) with integer class labels [0-9].
            augment (bool): If True, apply random augmentations on each access.
        """
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve a single sample, optionally augment, normalize, and convert to tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            image = self.augment_single(image)
            
        # normalize
        mean_norm = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std_norm = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
        image = (image - mean_norm) / std_norm

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def augment_single(self, image):
        """
        Apply random augmentations to a single CIFAR-10 image.

        Augmentations include random crop with padding, horizontal flip,
        brightness, contrast, saturation adjustments, and small rotations.

        Args:
            image (np.ndarray): Input image of shape (3, 32, 32).

        Returns:
            np.ndarray: Augmented image clipped to [0,1].
        """
        # Random crop with padding
        padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode='reflect')
        top = np.random.randint(0, 8)
        left = np.random.randint(0, 8)
        image = padded[:, top:top+32, left:left+32]

        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = image[:, :, ::-1]

        # Brightness
        image += np.random.uniform(-0.2, 0.2)

        # Contrast
        mean = image.mean(axis=(1, 2), keepdims=True)
        image = (image - mean) * np.random.uniform(0.8, 1.2) + mean

        # Saturation
        gray = image.mean(axis=0, keepdims=True)
        sat_factor = np.random.uniform(0.8, 1.2)
        image = image * sat_factor + gray * (1 - sat_factor)

        # Random rotation
        angle = np.random.uniform(-15, 15)
        for c in range(3):
            image[c] = rotate(image[c], angle, reshape=False, mode='reflect')

        return np.clip(image, 0.0, 1.0)

def unnormalize(tensor):
    """
    Reverse CIFAR-10 normalization for visualization.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (B, 3, H, W).

    Returns:
        torch.Tensor: Un-normalized image tensor.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean