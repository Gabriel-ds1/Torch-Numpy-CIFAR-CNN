"""
loss_functions.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Defines loss classes for training neural networks, including data loss calculations,
             regularization, and combined Softmax-CategoricalCrossentropy optimization.
Published: 04-26-2025
"""


from numpy_resnet.utils import backend
from .activation_functions import Activation_Softmax

# Common loss class
class Loss:
    """
    Base loss class providing regularization and accumulation logic.

    Methods:
        remember_trainable_layers: store layers for regularization.
        calculate: compute mean data loss (and optionally add regularization).
        calculate_accumulated: compute loss over multiple batches.
        new_pass: reset accumulation counters.
    """
    # Regularization loss calculation
    def regularization_loss(self):
        """
        Compute L1/L2 regularization penalty over stored trainable layers.

        Returns:
            float: total regularization loss.
        """
        # 0 by default
        regularization_loss = 0
        
        # Calculate regularization loss, iterate all trainable layers
        for layer in self.trainable_layers:

        #l1 regularization - weights, calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * backend.np.sum(backend.np.abs(layer.weights))

            #l2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * backend.np.sum(layer.weights * layer.weights)
            
            #l1 regularization - biases, calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * backend.np.sum(backend.np.abs(layer.biases))

            #l2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * backend.np.sum(layer.biases * layer.biases)
        return regularization_loss
    
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        """
        Store trainable layers for regularization.

        Args:
            trainable_layers (list): Layers with weights/bias attributes.
        """
        self.trainable_layers = trainable_layers
    
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):
        """
        Compute mean data loss over batch and optionally add regularization.

        Args:
            output (ndarray): Model predictions.
            y (ndarray): Ground-truth labels.
            include_regularization (bool): If True, return tuple (data_loss, reg_loss).

        Returns:
            float or (float, float): Data loss (and reg loss if requested).
        """
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = backend.np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += backend.np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss, self.regularization_loss()
    
    # calculates accumulated loss (across batches)
    def calculate_accumulated(self, *, include_regularization=False):
        """
        Compute mean loss over all accumulated batches.

        Args:
            include_regularization (bool): If True, include regularization term.

        Returns:
            float or (float, float): Mean data loss (and reg loss).
        """
        # calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # if just data loss - return it
        if not include_regularization:
            return data_loss
        
        # return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # reset variables for accumulated loss
    def new_pass(self):
        """
        Reset accumulators for a new epoch or validation pass.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
# Cross-Entropy Loss Class
class Loss_CategoricalCrossentropy(Loss):
    """
    Multi-class cross-entropy loss.

    forward: negative log likelihood for correct classes.
    backward: gradient w.r.t. predictions.
    """

    # Forward pass
    def forward(self, y_pred, y_true):
        """
        Compute sample-wise categorical crossentropy loss.

        Args:
            y_pred (ndarray): Predicted probabilities, shape (N, C).
            y_true (ndarray): True labels, either sparse (N,) or one-hot (N, C).

        Returns:
            ndarray: Loss per sample.
        """
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent dicision by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = backend.np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -> only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidence = backend.np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -backend.np.log(correct_confidence)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        """
        Backward pass for categorical crossentropy.

        Args:
            dvalues (ndarray): Predicted probabilities from softmax.
            y_true (ndarray): True labels.
        """
        # number of samples
        samples = len(dvalues)
        #number of labels in every sample, we'll use the first sample to count them
        labels = len(dvalues[0])

        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = backend.np.eye(labels)[y_true]
        
        # calculate gradient
        clipped_dvalues = backend.np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -y_true / clipped_dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Softmax_Loss_CatCrossentropy():
    """
    Combined Softmax activation plus categorical crossentropy loss optimization.

    Provides more efficient backward computation.
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        """
        Forward pass: softmax then crossentropy.

        Returns data loss.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        """
        Backward pass: gradient simplifies to (p - y_true) / N.
        """
        #number of samples
        samples = len(dvalues)
        # if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = backend.np.argmax(y_true, axis=1)

        # copy to safely modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    """
    Binary crossentropy for two-class classification or multi-label.
    """
    def forward(self, y_pred, y_true):
        """Compute BCE loss per sample."""
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = backend.np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * backend.np.log(y_pred_clipped) + (1 - y_true) * backend.np.log(1 - y_pred_clipped))
        sample_losses = backend.np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        """
        Backward pass for BCE.
        """
        # number of samples
        samples = len(dvalues)

        # number of outputs ine very sample, we'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = backend.np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss): # L2 loss
    """
    Mean squared error (L2) loss.
    """
    def forward(self, y_pred, y_true):
        """Compute MSE per sample."""
        # calculate loss
        sample_losses = backend.np.mean((y_true - y_pred)**2, axis=-1)

        # return losses
        return sample_losses
    
    def backward(self, dvalues, y_true):
        """Backward pass for MSE."""
        # number of samples
        samples = len(dvalues)

        #number of outputs in every sample, we'll use the first sample to count them
        outputs = len(dvalues[0])

        # gradients on values
        self.dinputs = -2 * (y_true - dvalues) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss): # L1 loss
    """
    Mean absolute error (L1) loss.
    """
    def forward(self, y_pred, y_true):
        """Compute MAE per sample."""
        #calculate loss
        sample_losses = backend.np.mean(backend.np.abs(y_true - y_pred), axis=-1)

        # return losses
        return sample_losses
    
    def backward(self, dvalues, y_true):
        """Backward pass for MAE."""
        # number of samples
        samples = len(dvalues)

        #number of outputs in every sample, we'll use the firs sample to count them
        outputs = len(dvalues[0])

        # calculate gradients
        self.dinputs = backend.np.sign(y_true - dvalues) / outputs
        #normalize gradient
        self.dinputs = self.dinputs / samples