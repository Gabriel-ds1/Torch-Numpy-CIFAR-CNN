"""
im2col.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Implements the im2col transformation for efficient convolution operations on 4D tensors
             by restructuring input patches into column vectors, enabling convolution via matrix multiplication.
Published: 04-26-2025
"""

from numpy_resnet.utils import backend
sliding_window_view = backend.np.lib.stride_tricks.sliding_window_view

def im2col(x, K, S, P):
    """
    Transform a batch of images into columnar patch representations for convolution.

    The im2col trick reshapes a 4D input tensor into a 2D matrix so that
    convolution can be performed as a single matrix multiplication.

    Args:
        x (backend.np.ndarray): Input tensor of shape (N, C, H, W), where
            N = batch size, C = channels, H = height, W = width.
        K (int): Kernel size (assumed square kernel of K x K).
        S (int): Stride length for sliding the kernel.
        P (int): Number of zero-padding pixels applied to height and width.

    Returns:
        X_col (backend.np.ndarray): 2D array of shape (C * K * K, N * H_out * W_out),
            where H_out = (H + 2*P - K)//S + 1 and W_out = (W + 2*P - K)//S + 1.

    Example:
        x = backend.np.random.rand(2, 3, 32, 32)
        cols = im2col(x, K=3, S=1, P=1)
        cols.shape
        (27, 2048)  # 3*3*3, 2*32*32
    """
    # Extract input dimensions
    N, C, H, W = x.shape
    # Compute output spatial dimensions
    H_out = (H + 2*P - K)//S + 1
    W_out = (W + 2*P - K)//S + 1

    # ===== Zero-padding =====
    # Create padded input with zeros around border
    H_p, W_p = H + 2*P, W + 2*P
    x_padded = backend.np.zeros((N, C, H_p, W_p), dtype=x.dtype)
    # Place original image in center
    x_padded[:, :, P:P+H, P:P+W] = x

    # ===== Extract sliding windows =====
    # Use numpy's stride tricks to get all KxK patches
    # windows shape: (N, C, H_padded-K+1, W_padded-K+1, K, K)
    sliding_window_view = backend.np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(x_padded, (K, K), axis=(2, 3)) # shape: (N, C, H_p-K+1, W_p-K+1, K, K)
    # Apply stride by slicing every S steps in height and width
    windows = windows[:, :, ::S, ::S, :, :]

    # ===== Rearrange and reshape =====
    # Transpose to bring patch dims and batch dims together:
    # from (N, C, H_out, W_out, K, K) to (C, K, K, N, H_out, W_out)
    windows = windows.transpose(1, 4, 5, 0, 2, 3)
    # Collapse to 2D: (C*K*K, N*H_out*W_out)
    return windows.reshape(C*K*K, N*H_out*W_out)