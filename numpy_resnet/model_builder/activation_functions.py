"""
activations.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Implements common activation functions (Linear, ReLU, Sigmoid, Softmax) using a NumPy-compatible backend.
             Each class provides forward and backward passes for neural network training, along with prediction methods.
Published: 04-26-2025
"""

from numpy_resnet.utils import backend

class Activation_Linear:
    """
    Linear (identity) activation function.
    Useful as a no-op activation or for regression outputs.

    forward: stores inputs and outputs same values.
    backward: gradient w.r.t. inputs is passed through unchanged.
    predictions: returns raw outputs.
    """
    # Linear activation technically does nothing, but we will use in our regression model for clarity
    def forward(self, inputs, training):
        """
        Forward pass for linear activation.

        Args:
            inputs (backend.np.ndarray): Input data, any shape.
            training (bool): Ignored, present for API consistency.
        """
        # Store inputs and set output equal to inputs
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        """
        Backward pass for linear activation.

        Args:
            dvalues (backend.np.ndarray): Upstream gradients of same shape as inputs.
        """
        # Derivative is 1. 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        """
        Return predictions for linear output (identity).

        Args:
            outputs (backend.np.ndarray): Activation outputs.

        Returns:
            backend.np.ndarray: Raw outputs.
        """
        return outputs

class Activation_ReLU:
    """
    Rectified Linear Unit (ReLU) activation.
    Sets negative inputs to zero; passes positives unchanged.

    forward: apply elementwise max(0, x).
    backward: zero out gradients where input was <= 0.
    predictions: return raw outputs for subsequent processing.
    """
    def forward(self, inputs, training):
        """
        Forward pass for ReLU activation.

        Args:
            inputs (backend.np.ndarray): Input data.
            training (bool): Ignored.
        """
        self.inputs = inputs
        self.output = backend.np.maximum(0, inputs)  # ReLU activation function

    def backward(self, dvalues):
        """
        Backward pass for ReLU.

        Args:
            dvalues (backend.np.ndarray): Upstream gradients.
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 #derivative of relu

    # Calculate predictions for outputs
    def predictions(self, outputs):
        """
        For classification tasks, ReLU outputs used directly.

        Args:
            outputs (backend.np.ndarray): Activation outputs.

        Returns:
            backend.np.ndarray: Same as outputs.
        """
        return outputs


# Softmax Activation Function
class Activation_Softmax:
    """
    Softmax activation function for multi-class outputs.
    Converts logits into probability distributions.

    forward: exponentiate and normalize across classes.
    backward: compute gradients via Jacobian matrix per sample.
    predictions: choose class with highest probability.
    """
    def forward(self, inputs, training):
        """
        Forward pass for Softmax activation.

        Args:
            inputs (backend.np.ndarray): Logits array of shape (N, classes).
            training (bool): Ignored.
        """
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = backend.np.exp(inputs - backend.np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / backend.np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        """
        Backward pass for Softmax.
        Computes gradient of loss w.r.t. inputs using the Jacobian.

        Args:
            dvalues (backend.np.ndarray): Upstream gradients, same shape as output.
        """
        # Create uninitialized array
        self.dinputs = backend.np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = backend.np.diagflat(single_output) - backend.np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = backend.np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        """
        Return class indices corresponding to highest probability.

        Args:
            outputs (backend.np.ndarray): Softmax probabilities.

        Returns:
            backend.np.ndarray: Predicted class labels.
        """
        return backend.np.argmax(outputs, axis=1)

# Sigmoid activation
class Activation_Sigmoid:
    """
    Sigmoid activation function.
    Outputs values in (0,1); used for binary classification.

    forward: apply logistic function.
    backward: gradient via output * (1 - output).
    predictions: threshold at 0.5.
    """
    # forward pass
    def forward(self, inputs, training):
        """
        Forward pass for Sigmoid activation.

        Args:
            inputs (backend.np.ndarray): Input data.
            training (bool): Ignored.
        """
        # Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + backend.np.exp(-inputs))
    
    # Backward pass
    def backward(self, dvalues):
        """
        Backward pass for Sigmoid.

        Args:
            dvalues (backend.np.ndarray): Upstream gradients.
        """
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        """
        Convert probabilities to binary class predictions (0 or 1).

        Args:
            outputs (backend.np.ndarray): Activation outputs.

        Returns:
            backend.np.ndarray: Binary predictions.
        """
        return (outputs > 0.5) * 1