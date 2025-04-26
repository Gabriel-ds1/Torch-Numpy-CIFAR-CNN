"""
gradient_check.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Provides gradient checking utilities for convolutional layers,
             comparing analytical gradients (from backward pass) to numerical approximations.
             Includes implementations for custom NumPy-based Conv2D layers and PyTorch's nn.Conv2d.
Published: 04-26-2025
"""

import numpy as np
import torch
import torch.nn as nn
from numpy_resnet.model_builder.layers import Layer_Conv2D

def gradient_check_conv_numpy(layer, input_data, target, epsilon=1e-5):
    """
    Perform gradient check on a custom Conv2D layer using finite differences.

    Compares the analytical gradient computed during the backward pass
    with a numerical gradient approximation for a single randomly chosen weight.

    Args:
        layer (Layer_Conv2D): Custom convolutional layer instance.
        input_data (np.ndarray): Input tensor of shape (N, C_in, H, W).
        target (np.ndarray): Target gradient array of same shape as layer output.
        epsilon (float): Small perturbation for finite differences.
    """
    # Forward pass: compute output and dummy loss
    layer.forward(input_data, training=True)
    # Use simple dot-product loss: sum(output * target)
    loss = np.sum(layer.output * target)  # dummy loss: elementwise dot

    # Backward pass: set doutput = target -> dloss/doutput
    d_out = target  # since d(loss)/d(output) = target
    layer.backward(d_out)

    # Randomly select one weight index for testing
    c_out, c_in, i, j = np.random.randint(0, layer.weights.shape[0]), \
                        np.random.randint(0, layer.weights.shape[1]), \
                        np.random.randint(0, layer.weights.shape[2]), \
                        np.random.randint(0, layer.weights.shape[3])

    # Save original value
    orig_value = layer.weights[c_out, c_in, i, j]

    # Numerical gradient: perturb + epsilon
    layer.weights[c_out, c_in, i, j] = orig_value + epsilon
    layer.forward(input_data, training=False)
    loss_plus = np.sum(layer.output * target)

    # Perturb - epsilon
    layer.weights[c_out, c_in, i, j] = orig_value - epsilon
    layer.forward(input_data, training=False)
    loss_minus = np.sum(layer.output * target)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore original weight
    layer.weights[c_out, c_in, i, j] = orig_value

    # Analytical gradient from backward pass
    analytical_grad = layer.dweights[c_out, c_in, i, j]

    # Compute relative error
    rel_error = abs(numerical_grad - analytical_grad) / max(1e-8, abs(numerical_grad) + abs(analytical_grad))

    # Print results
    print(f"Numerical grad: {numerical_grad:.6f}")
    print(f"Analytical grad: {analytical_grad:.6f}")
    print(f"Relative error: {rel_error:.6e}")


def gradient_check_conv_torch(layer, input_data, target, epsilon=1e-5):
    """
    Perform gradient check on a PyTorch Conv2d layer using finite differences.

    Compares the analytical gradient from PyTorch autograd
    to a numerical approximation for one randomly chosen weight.

    Args:
        layer (nn.Conv2d): PyTorch convolutional layer.
        input_data (torch.Tensor): Input tensor requiring grad.
        target (torch.Tensor): Target tensor same shape as layer output.
        epsilon (float): Perturbation size for numerical grad.
    """
    # Zero gradients and enable grad tracking
    layer.zero_grad()
    input_data.requires_grad = True

    # Forward pass and backward to compute analytical gradient
    output = layer(input_data)
    loss = torch.sum(output * target)  # dummy loss
    loss.backward()

    # Choose random weight index
    c_out, c_in, i, j = torch.randint(layer.weight.shape[0], (1,)).item(), \
                        torch.randint(layer.weight.shape[1], (1,)).item(), \
                        torch.randint(layer.weight.shape[2], (1,)).item(), \
                        torch.randint(layer.weight.shape[3], (1,)).item()

    # Save original weight value
    orig_value = layer.weight.data[c_out, c_in, i, j].item()

    # Numerical gradient: + epsilon
    layer.weight.data[c_out, c_in, i, j] = orig_value + epsilon
    output = layer(input_data)
    loss_plus = torch.sum(output * target).item()

    # Numerical gradient: - epsilon
    layer.weight.data[c_out, c_in, i, j] = orig_value - epsilon
    output = layer(input_data)
    loss_minus = torch.sum(output * target).item()

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore original weight
    layer.weight.data[c_out, c_in, i, j] = orig_value

    # Analytical gradient from autograd
    analytical_grad = layer.weight.grad[c_out, c_in, i, j].item()

    # Relative error computation
    rel_error = abs(numerical_grad - analytical_grad) / max(1e-8, abs(numerical_grad) + abs(analytical_grad))

    # Print results
    print(f"Numerical grad: {numerical_grad:.6f}")
    print(f"Analytical grad: {analytical_grad:.6f}")
    print(f"Relative error: {rel_error:.6e}")


# set up numpy layer and data
numpy_layer = Layer_Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, use_im2col=True)
numpy_input_data = np.random.randn(1, 1, 5, 5)
numpy_target = np.random.randn(1, 1, 5, 5)
# calculate numpy conv gradients
gradient_check_conv_numpy(numpy_layer, numpy_input_data, numpy_target)


# Set up PyTorch layer and data
torch_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
torch_input_data = torch.randn(1, 1, 5, 5, dtype=torch.double, requires_grad=True)
torch_target = torch.randn(1, 1, 5, 5, dtype=torch.double)

# Convert layer to double for numerical stability
torch_layer = torch_layer.double()
# calculate PyTorch conv gradients
gradient_check_conv_torch(torch_layer, torch_input_data, torch_target)