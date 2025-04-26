"""
accuracy.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Provides accuracy metrics for regression and classification tasks, with support for
             batch-level and epoch-level accumulation. Users implement domain-specific logic
             by subclassing `Accuracy` and overriding the `compare` method.
Published: 04-26-2025
"""


from numpy_resnet.utils import backend

# Common accuracy class
class Accuracy:
    """
    Base class for accuracy metrics with accumulation support.

    Subclasses must implement:
        init(y): optional initialization based on ground truth.
        compare(predictions, y): return boolean array of correct predictions.

    Methods:
        calculate: compute accuracy for a batch and update accumulators.
        calculate_accumulated: compute accuracy over all accumulated batches.
        new_pass: reset accumulators for a new epoch or evaluation.
    """
    # Calculates an accuracy
    # Given predictions and ground truth values
    def calculate(self, predictions, y):
        """
        Calculate accuracy for a single batch and update accumulators.

        Args:
            predictions (ndarray): Predicted values or labels.
            y (ndarray): True values or labels.

        Returns:
            float: Batch accuracy (fraction correct).
        """
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = backend.np.mean(comparisons)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += backend.np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # return accuracy
        return accuracy
    
    # Calculates accumulated accuracy (across batches)
    def calculate_accumulated(self):
        """
        Compute accuracy across all previously accumulated batches.

        Returns:
            float: Epoch-level accuracy.
        """
        # calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    # Reset variables for accumulated accuracy
    def new_pass(self):
        """
        Reset accumulators at the start of a new epoch or evaluation.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    """
    Accuracy metric for regression tasks: counts predictions within a tolerance.

    Attributes:
        precision (float): Acceptable error margin for a prediction to be considered correct.
    """
    def __init__(self):
        # Create precision property
        self.precision = None
    
    # Calculate precision value based on passed-in ground truth
    def init(self, y, reinint=False):
        """
        Initialize tolerance based on true values' standard deviation.

        Args:
            y (ndarray): Ground truth values.
            reinint (bool): If True, recompute precision even if already set.
        """
        if self.precision is None or reinint:
            self.precision = backend.np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        """
        Compare regression outputs: True if absolute error < precision.

        Args:
            predictions (ndarray): Predicted continuous values.
            y (ndarray): True continuous values.

        Returns:
            ndarray of bool: Correctness mask.
        """
        return backend.np.abs(predictions - y) < self.precision
    
# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    """
    Accuracy metric for classification tasks.

    Attributes:
        binary (bool): If True, treats outputs as binary classification.
    """
    def __init__(self, *, binary=False):
        """
        No initialization needed for categorical accuracy.
        """
        # Binary mode?
        self.binary = binary

    # no initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        """
        Compare classification outputs: matches predicted vs true labels.

        Converts one-hot y to class indices when needed.

        Args:
            predictions (ndarray): Predicted class indices.
            y (ndarray): True labels, either sparse (N,) or one-hot encoded (N,C).

        Returns:
            ndarray of bool: Correctness mask.
        """
        # If one-hot encoded and not binary, convert to indices
        if not self.binary and len(y.shape) == 2:
            y = backend.np.argmax(y, axis=1)
        return predictions == y