"""
optimizers.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Implements optimization algorithms (SGD, Adagrad, RMSprop, Adam, Adamax) with support
             for learning rate scheduling, momentum, and parameter-specific state caching.
Published: 04-26-2025
"""

from numpy_resnet.utils import backend

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum and learning rate scheduling.

    Attributes:
        learning_rate (float): Base learning rate.
        schedule (callable or None): Learning rate schedule function taking epoch -> lr.
        momentum (float): Momentum factor (0 for vanilla SGD).
    """
    # initialize optimizer - set settings
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=1., schedule=None, momentum=0.):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate (float): Initial learning rate.
            schedule (callable): Function mapping epoch index to lr, or None.
            momentum (float): Momentum coefficient (0 disables momentum).
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.schedule = schedule
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self, epoch):
        """
        Adjust current_learning_rate based on schedule before parameter update.

        Args:
            epoch (int): Current epoch number (0-based).
        """
        if self.schedule:
            self.current_learning_rate = self.schedule(epoch)
    
    # update parameters
    def update_params(self, layer):
        """
        Apply parameter updates to a single layer.

        Supports momentum when self.momentum > 0, otherwise vanilla SGD.
        """
        # if we use momentum
        if self.momentum:
            # if layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = backend.np.zeros_like(layer.weights) #backend.np.zeros_like -> create zeros array with same shape like (layers.weights)
                #if there is no momentum array for weights the array doesnt exist for biases yet either
                layer.bias_momentums = backend.np.zeros_like(layer.biases)

            #build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else: # vanilla SGD updates without momentum
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Increment iteration counter after update.
        """
        self.iterations += 1

class Optimizer_Adagrad:
    """
    Adagrad optimizer: adaptive learning rates per parameter based on historical gradients.

    Attributes:
        learning_rate (float): Base learning rate.
        epsilon (float): Smoothing term to avoid division by zero.
        schedule (callable or None): Learning rate schedule.
    """
    # initialize optimizer - set settings
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=1., schedule=None, epsilon=1e-7):
        """
        Initialize Adagrad optimizer.

        Args:
            learning_rate (float): Initial learning rate.
            schedule (callable): Optional LR schedule.
            epsilon (float): Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.schedule = schedule
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self, epoch):
        if self.schedule:
            self.current_learning_rate = self.schedule(epoch)
    
    # update parameters
    def update_params(self, layer):
        """
        Update layer parameters using Adagrad rule.
        """

        # if layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = backend.np.zeros_like(layer.weights) #backend.np.zeros_like -> create zeros array with same shape like (layers.weights)

            layer.bias_cache = backend.np.zeros_like(layer.biases)

        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (backend.np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (backend.np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    """
    RMSprop optimizer: maintains moving average of squared gradients (rho factor).
    """
    # initialize optimizer - set settings
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=0.001, schedule=None, epsilon=1e-7, rho=0.9):
        """
        Args:
            learning_rate (float): Base learning rate.
            schedule (callable): LR schedule function.
            epsilon (float): Numerical stability constant.
            rho (float): Decay rate for moving average.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.schedule = schedule
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho # the cache memory decay rate

    # Call once before any parameter updates
    def pre_update_params(self, epoch):
        if self.schedule:
            self.current_learning_rate = self.schedule(epoch)
    
    # update parameters
    def update_params(self, layer):
        """
        Update layer parameters with RMSprop.
        """

        # if layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = backend.np.zeros_like(layer.weights) #backend.np.zeros_like -> create zeros array with same shape like (layers.weights)

            layer.bias_cache = backend.np.zeros_like(layer.biases)

        #Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (backend.np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (backend.np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    """
    Adam optimizer: combines momentum and RMSprop with bias correction.
    """
    # initialize optimizer - set settings
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=0.001, schedule=None, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """
        Args:
            learning_rate (float): Base learning rate.
            schedule (callable): LR schedule.
            epsilon (float): Stability term.
            beta_1 (float): Momentum decay.
            beta_2 (float): RMS decay.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.schedule = schedule
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self, epoch):
        if self.schedule:
            self.current_learning_rate = self.schedule(epoch)
    
    # update parameters
    def update_params(self, layer):
        """
        Update layer parameters with Adam algorithm.
        """

        # if layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = backend.np.zeros_like(layer.weights)
            layer.weight_cache = backend.np.zeros_like(layer.weights) #backend.np.zeros_like -> create zeros array with same shape like (layers.weights)
            layer.bias_momentums = backend.np.zeros_like(layer.biases)
            layer.bias_cache = backend.np.zeros_like(layer.biases)

        #Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # get corrected momentum, self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (backend.np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (backend.np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adamax:
    """
    Adamax optimizer: variant of Adam using infinity norm for scaling.
    """
    # initialize optimizer - set settings
    # learning rate default set to 0.002 as per original Adamax paper
    def __init__(self, learning_rate=0.002, schedule=None, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.schedule = schedule
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self, epoch):
        if self.schedule:
            self.current_learning_rate = self.schedule(epoch)

    # update parameters
    def update_params(self, layer):
        """
        Update using Adamax: uses max of past |grad| for normalization.
        """

        # if layer does not contain momentums or inf‐norm arrays, create them
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = backend.np.zeros_like(layer.weights)
            layer.weight_inf_norm = backend.np.zeros_like(layer.weights)
            layer.bias_momentums = backend.np.zeros_like(layer.biases)
            layer.bias_inf_norm = backend.np.zeros_like(layer.biases)

        # Update biased first moment estimate
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums
            + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums
            + (1 - self.beta_1) * layer.dbiases
        )

        # Compute the exponentially weighted infinity norm (max)
        layer.weight_inf_norm = backend.np.maximum(
            self.beta_2 * layer.weight_inf_norm,
            backend.np.abs(layer.dweights)
        )
        layer.bias_inf_norm = backend.np.maximum(
            self.beta_2 * layer.bias_inf_norm,
            backend.np.abs(layer.dbiases)
        )

        # Bias‐corrected first moment
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        # Parameter update (no bias‐correction on the inf‐norm term)
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / (layer.weight_inf_norm + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / (layer.bias_inf_norm + self.epsilon)

    def post_update_params(self):
        self.iterations += 1