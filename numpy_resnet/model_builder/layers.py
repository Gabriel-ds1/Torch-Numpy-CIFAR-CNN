"""
layers.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Implements neural network building blocks: Dense, Input, Dropout, Conv2D, BatchNorm2d,
             Pooling, AdaptiveAvgPool2d, ResidualBlock, and Reshape layers, all using a NumPy/CuPy backend.
             Each layer provides forward and backward propagation with optional regularization and im2col acceleration.
Published: 04-26-2025
"""

from numpy_resnet.utils import backend
from numpy_resnet.utils.vectorize import im2col

# Dense Layer Class
class Layer_Dense:
    """
    Fully connected (dense) neural network layer.

    Applies linear transformation: output = inputs @ weights + biases.
    Supports L1/L2 regularization on weights and biases.
    """
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, init_type="he_scaling", weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        """
        Initialize weights and biases.

        Args:
            n_inputs (int): Number of input features.
            n_neurons (int): Number of output neurons.
            init_type (str): 'he_scaling', 'xavier', or other for random init.
            weight_regularizer_l1 (float): L1 reg coefficient on weights.
            weight_regularizer_l2 (float): L2 reg coefficient on weights.
            bias_regularizer_l1 (float): L1 reg coefficient on biases.
            bias_regularizer_l2 (float): L2 reg coefficient on biases.
        """
        self.init_type = init_type
        # Initialize weights
        if self.init_type == "he_scaling":
            self.weights = backend.np.random.randn(n_inputs, n_neurons) * backend.np.sqrt(2. / n_inputs)
        elif self.init_type == "xavier":
            self.weights = backend.np.random.randn(n_inputs, n_neurons) * backend.np.sqrt(1. / n_inputs)
        else:
            self.weights = 0.1 * backend.np.random.randn(n_inputs, n_neurons)
        # Initialize biases
        self.biases = backend.np.zeros((1, n_neurons))

        # set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # retrieve layer parameters
    def get_parameters(self):
        """
        Retrieve current weights and biases.

        Returns:
            tuple: (weights, biases)
        """
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        """
        Set layer weights and biases (used for loading models).

        Args:
            weights (ndarray): New weight matrix.
            biases (ndarray): New bias vector.
        """
        self.weights = weights
        self.biases = biases

    # Forward pass
    def forward(self, inputs, training):
        """
        Forward pass: linear transform.

        Args:
            inputs (ndarray): Input batch of shape (N, n_inputs).
            training (bool): Ignored (for API consistency).

        Returns:
            ndarray: Output batch of shape (N, n_neurons).
        """
        self.inputs = inputs
        # Calculate output of the layer
        self.output = backend.np.dot(inputs, self.weights) + self.biases
        return self.output

    # backward pass
    def backward(self, dvalues):
        """
        Backward pass: gradients w.r.t inputs, weights, and biases.

        Applies L1/L2 regularization adjustments.

        Args:
            dvalues (ndarray): Gradient of loss w.r.t. layer outputs.

        Returns:
            ndarray: Gradient of loss w.r.t. layer inputs.
        """
        # gradients on parameters
        self.dweights = backend.np.dot(self.inputs.T, dvalues)
        self.dbiases = backend.np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = backend.np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = backend.np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        #L2 on weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradient on values
        self.dinputs = backend.np.dot(dvalues, self.weights.T)
        return self.dinputs

class Layer_Input:
    """
    Input placeholder layer: passes inputs through.
    """
    def forward(self, inputs, training):
        """Store and output inputs unmodified."""
        self.output=inputs

class Layer_Dropout:
    """
    Dropout layer for regularization.

    Randomly masks inputs during training to prevent overfitting.
    Uses inverted dropout scaling.
    """
    def __init__(self, rate):
        """
        Args:
            rate (float): Fraction of units to drop (0<rate<1).
        """
        # Keep probability
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        Forward pass: apply dropout mask if training.
        """
        # save input values
        self.inputs = inputs

        # if not in training mode - dont use dropout, return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        # below is an example of inverted dropout, it slightly scales every neuron that's kept activated to compensate for the zeroing out of random neurons (less total weight)
        self.binary_mask = backend.np.random.binomial(1, self.rate, size=inputs.shape) / self.rate # ^ is why we divide by self.rate (e.g. 0.1, 1 is now 1.11111 all across the board.)
        # Apply mask to output values
        self.output = inputs * self.binary_mask # either zero out or add .1111 (x * 1.111) to original value
        return self.output
    
    def backward(self, dvalues):
        """
        Backward pass: mask gradients.
        """
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

class Layer_Conv2D:
    """
    2D convolutional layer with optional im2col acceleration.
    Supports L1/L2 regularization on filters and biases.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_type="he_scaling",  weight_regularizer_l1=0, 
                 weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0, use_im2col=False):
        """
        Initialize filters and biases.

        Args:
            in_channels (int): Number of input feature maps.
            out_channels (int): Number of output feature maps.
            kernel_size (int): Height/width of square filter.
            stride (int): Stride of convolution.
            padding (int): Zero-padding on each side.
            init_type (str): Weight init strategy.
            use_im2col (bool): Whether to use im2col matrix multiply.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_type = init_type
        self.use_im2col = use_im2col

        # He initialization
        if self.init_type == "he_scaling":
            self.weights = backend.np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * backend.np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        elif self.init_type == "xavier":
            self.weights = backend.np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * backend.np.sqrt(1. / (in_channels * kernel_size * kernel_size))
        else:
            self.weights = 0.1 * backend.np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = backend.np.zeros(out_channels)

        # set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training=True):
        """
        Forward pass for convolution.

        Args:
            inputs (ndarray): Input batch shape (N, C, H, W).
            training (bool): Ignored for conv.

        Returns:
            ndarray: Output feature maps (N, out_channels, H_out, W_out).
        """
        # x: (N, in_ch, H, W)
        # return out: (N, out_ch, H_out, W_out)
        # also save im2col indices or patches for backward
        self.inputs = inputs
        N, C, H, W = inputs.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # compute output spatial dims
        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        # im2col (reshape 4D tensor to 2D matrix to leverage highly optimized matrix‑multiply)
        if self.use_im2col:
            X_col = im2col(inputs, K, S, P)
            W_col = self.weights.reshape(self.out_channels, -1)
            out = W_col @ X_col + self.biases[:, None]
            out = out.reshape(self.out_channels, N, H_out, W_out).transpose(1, 0, 2, 3)
        else:
            # Naive convolution loops with padding
            padded = backend.np.pad(inputs, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
            out = backend.np.zeros((N, self.out_channels, H_out, W_out))

            for n in range(N):
                for c_out in range(self.out_channels):
                    for i in range(H_out):
                        for j in range(W_out):
                            vert_start = i * S
                            horiz_start = j * S
                            patch = padded[n, :, vert_start:vert_start+K, horiz_start:horiz_start+K]
                            out[n, c_out, i, j] = backend.np.sum(patch * self.weights[c_out]) + self.biases[c_out]
        
        # cache for backward
        if self.use_im2col:
            self.cache = (inputs, X_col, W_col, out.shape)
        self.output = out
        return self.output

    def backward(self, dvalue):
        """
        Backward pass: dispatch to im2col or direct method.
        """
        if self.use_im2col:
            self.dinputs = self._backward_im2col(dvalue)
        else:
            self.dinputs = self._backward_direct(dvalue)

    def _backward_im2col(self, dvalue):
        """
        Efficient backward pass using im2col representation.
        """
        x, X_col, W_col, out_shape = self.cache
        N, C_in, H_in, W_in = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        # Pull out output spatial dims
        H_out, W_out = out_shape[2], out_shape[3]

        # reshape dvalue into columns
        dvalue_col = dvalue.transpose(1,0,2,3).reshape(self.out_channels, -1)

        # grad w.r.t weights
        dW = dvalue_col @ X_col.T
        self.dweights = dW.reshape(self.weights.shape)

        # grad w.r.t bias
        self.dbiases = backend.np.sum(dvalue_col, axis=1)

        # grad w.r.t X_col
        dX_col = W_col.T @ dvalue_col # shape (C_in*K*K, N*H_out*W_out)

        # col2im: scatter patches back into the padded input gradient
        dx_padded = backend.np.zeros((N, C_in, H_in + 2*P, W_in + 2*P), dtype=x.dtype)
        dcols = dX_col.reshape(C_in, K, K, N, H_out, W_out)
        for i in range(K):
            for j in range(K):
                patch_grad = dcols[:, i, j, :, :, :].transpose(1, 0, 2, 3)
                dx_padded[:, :, 
                      i : i + S*H_out : S, 
                      j : j + S*W_out : S] += patch_grad
        # unpad
        if P > 0:
            d_x = dx_padded[:, :, P:-P, P:-P]
        else:
            d_x = dx_padded

        # Apply L1 and L2 regularization to weights
        if self.weight_regularizer_l1 > 0:
            self.dweights += self.weight_regularizer_l1 * backend.np.sign(self.weights)

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # Apply L1 and L2 regularization to biases
        if self.bias_regularizer_l1 > 0:
            self.dbiases += self.bias_regularizer_l1 * backend.np.sign(self.biases)

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = d_x
        return self.dinputs
    
    def _backward_direct(self, dvalue):
        """
        Naive backward pass with loops.
        """
        N, C_in, H_in, W_in = self.inputs.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        H_out, W_out = dvalue.shape[2], dvalue.shape[3]

        # Pad input and gradient
        padded_input = backend.np.pad(self.inputs, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        padded_dinputs = backend.np.zeros_like(padded_input)

        # Initialize gradients
        self.dweights = backend.np.zeros_like(self.weights)
        self.dbiases = backend.np.zeros_like(self.biases)

        for n in range(N):
            for c_out in range(self.out_channels):
                self.dbiases[c_out] += backend.np.sum(dvalue[n, c_out])
                for i in range(H_out):
                    for j in range(W_out):
                        vert_start = i * S
                        horiz_start = j * S

                        patch = padded_input[n, :, vert_start:vert_start+K, horiz_start:horiz_start+K]

                        # Grad w.r.t. weights
                        self.dweights[c_out] += patch * dvalue[n, c_out, i, j]

                        # Grad w.r.t. inputs
                        padded_dinputs[n, :, vert_start:vert_start+K, horiz_start:horiz_start+K] += self.weights[c_out] * dvalue[n, c_out, i, j]

        # Remove padding from dinputs
        if P > 0:
            self.dinputs = padded_dinputs[:, :, P:-P, P:-P]
        else:
            self.dinputs = padded_dinputs
        
        # Apply L1 and L2 regularization to weights
        if self.weight_regularizer_l1 > 0:
            self.dweights += self.weight_regularizer_l1 * backend.np.sign(self.weights)

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # Apply L1 and L2 regularization to biases
        if self.bias_regularizer_l1 > 0:
            self.dbiases += self.bias_regularizer_l1 * backend.np.sign(self.biases)

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        return self.dinputs

class Layer_BatchNorm2d:
    """
    2D batch normalization for convolutional layers.

    Normalizes across batch, height, and width dimensions per channel.
    Maintains running estimates for inference.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features (int): Number of channels.
            eps (float): Small constant for numeric stability.
            momentum (float): Momentum for running mean/variance.
        """
        # parameters, plus running_mean, running_var
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # learnable parameters
        self.gamma = backend.np.ones((num_features,))
        self.beta = backend.np.zeros((num_features,))
        # running stats
        self.running_mean = backend.np.zeros((num_features,))
        self.running_var = backend.np.ones((num_features,))
        # cache for backward
        self.cache = None

    def forward(self, x, training=True):
        """
        Forward pass for batchnorm.

        Args:
            x (ndarray): Input shape (N,C,H,W).
            training (bool): Use batch stats vs running stats.

        Returns:
            ndarray: Normalized and scaled output.
        """
        # if training, compute batch mean/var; update running stats
        # save normalized x and var for backward
        N, C, H, W = x.shape
        # compute mean/var per channel
        if training:
            # axes: 0,2,3
            batch_mean = backend.np.mean(x, axis=(0, 2, 3))
            batch_var = backend.np.var(x, axis=(0, 2, 3))

            # update running stats
            self.running_mean = (self.momentum * batch_mean + (1 - self.momentum) * self.running_mean)
            self.running_var = (self.momentum * batch_var + (1 - self.momentum) * self.running_var)
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Now we noremalize
        x_centered = x - mean[None, :, None, None]
        std = backend.np.sqrt(var[None, :, None, None] + self.eps)
        x_norm = x_centered / std
        # gamma and beta allows the model to shift normalization (can even completely undo it) so that it achieves best results
        self.output = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

        # cache values for backward pass
        self.cache = (x_norm, std, x_centered)
        return self.output
    
    def backward(self, dvalue):
        """
        Backward pass for BatchNorm2d.
        Args:
            dvalue: gradient w.r.t. output, shape (N,C,H,W)
        Returns:
            d_x: gradient w.r.t. input x
            also stores d_gamma and d_beta
        """
        x_norm, std, x_centered = self.cache
        N, C, H, W = dvalue.shape
        total_elements = N * H * W

        # gradient w.r.t gamma and beta
        d_gamma = backend.np.sum(dvalue * x_norm, axis=(0, 2, 3))
        d_beta = backend.np.sum(dvalue, axis=(0, 2, 3))

        # gradient w.r.t x_norm
        dx_norm = dvalue * self.gamma[None, :, None, None]
        # gradient w.r.t vairance
        d_var = backend.np.sum(dx_norm * x_centered * (-0.5) * std**(-3), axis=(0, 2, 3))
        # grad w.r.t. mean
        d_mean = backend.np.sum(dx_norm * (-1/std), axis=(0,2,3)) + d_var * backend.np.sum(-2 * x_centered, axis=(0,2,3)) / total_elements

        # gradient w.r.t x (input)
        d_x = (dx_norm / std + d_var[None,:,None,None] * 2 * x_centered / total_elements + d_mean[None,:,None,None] / total_elements)

        # store parameter gradients
        self.d_gamma = d_gamma
        self.d_beta = d_beta
        self.dinputs = d_x
        return self.dinputs
    
class Layer_MaxPool2d:
    """
    2D max pooling.

    Extracts maximum value over non-overlapping windows.
    """
    def __init__(self, kernel_size, stride=None):
        """
        Args:
            kernel_size (int): Size of pooling window.
            stride (int): Step size; defaults to kernel_size.
        """
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.cache = None

    def forward(self, x, training):
        """
        Forward pass for max pooling.

        Args:
            x (ndarray): Input shape (N,C,H,W).
        """
        N, C, H, W = x.shape
        K, S = self.kernel_size, self.stride
        H_out = (H - K)//S + 1
        W_out = (W - K)//S + 1
        out = backend.np.zeros((N, C, H_out, W_out))
        # indices of maxima for backward
        max_idx = backend.np.zeros_like(out, dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h0, w0 = i*S, j*S
                        window = x[n, c, h0:h0+K, w0:w0+K]
                        out[n, c, i, j] = backend.np.max(window)
                        max_idx[n, c, i, j] = backend.np.argmax(window)

        self.cache = (x.shape, max_idx)
        self.output = out
        return self.output
    
    def backward(self, dvalue):
        """
        Backward pass for max pooling.

        Routes gradient to position of max in each pooling window.
        """
        x_shape, max_idx = self.cache
        N, C, H, W = x_shape
        K, S = self.kernel_size, self.stride
        H_out, W_out = dvalue.shape[2:]
        # 1) Turn each scalar grad dvalue[n,c,i,j] into a one-hot over the K*K patch:
        #    shape: (N, C, H_out, W_out, K*K)
        one_hot = backend.np.eye(K*K, dtype=dvalue.dtype)[max_idx]

        # 2) Multiply by dvalue[...,None] to get per-patch gradients:
        #    shape (N, C, H_out, W_out, K*K)
        d_patches = dvalue[..., None] * one_hot

        # 3) Reshape into (N, C, K, K, H_out, W_out) so we can scatter channels at once:
        #    note the transpose just reorders dims
        d_patches = (
            d_patches
            .reshape(N, C, H_out, W_out, K, K)
            .transpose(0, 1, 4, 5, 2, 3)
        )

        # 4) Allocate full-sized grad and scatter each (i,j) patch with slicing:
        d_x = backend.np.zeros((N, C, H, W), dtype=dvalue.dtype)
        for i in range(K):
            for j in range(K):
                # each (i,j) slice steps by stride S
                d_x[:, :, i : i + S*H_out : S, j : j + S*W_out : S] += d_patches[:, :, i, j]
        self.dinputs = d_x
        return self.dinputs
    
class Layer_AdaptiveAvgPool2d:
    """
    2D adaptive average pooling layer to a fixed output size (H_out, W_out).
    Currently only supports output_size=(1,1) for global pooling.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple) and len(output_size) == 2, \
            "output_size must be a tuple (H_out, W_out)"
        self.output_size = output_size
        self.cache_input_shape = None

    def forward(self, x, training):
        """
        Forward pass: global average pool if output_size==(1,1).
        """
        N, C, H, W = x.shape
        H_out, W_out = self.output_size
        assert (H_out, W_out) == (1, 1), "Only output_size=(1,1) is supported currently"

        # global average pooling
        out = backend.np.mean(x, axis=(2, 3)).reshape(N, C, 1, 1)
        self.cache_input_shape = x.shape
        self.output = out
        return self.output

    def backward(self, dvalue):
        """
        Backward pass: distribute gradient evenly across input spatial dims.
        """
        N, C, H, W = self.cache_input_shape
        # each input position gets equal share of the gradient
        d_x = (dvalue / float(H * W)) * backend.np.ones((N, C, H, W))
        self.dinputs = d_x
        return self.dinputs

class Layer_ResidualBlock:
    """
    Basic ResNet residual block: two conv→bn→dropout sequences with skip connection.
    """
    def __init__(self, channels, dropout_rate=0.2):
        # conv → bn → dropout → relu  → conv → bn → dropout → add → relu
        self.conv1 = Layer_Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn1 = Layer_BatchNorm2d(channels)
        self.dropout1 = Layer_Dropout(dropout_rate)
        self.conv2 = Layer_Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn2 = Layer_BatchNorm2d(channels)
        self.dropout2 = Layer_Dropout(dropout_rate)
        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass through two branches + skip connection with two ReLUs.
        """
        # first branch
        self.cache['x_skip'] = x

        out = self.conv1.forward(x,     training=training)
        out = self.bn1.forward(out,      training=training)
        out = self.dropout1.forward(out, training=training)
        # dropout may return None in inference—pull from .output
        if out is None:
            out = self.dropout1.output
        # record ReLU mask #1
        m1 = (out > 0)
        out = out * m1
        self.cache['mask1'] = m1

        # second branch
        out = self.conv2.forward(out,     training=training)
        out = self.bn2.forward(out,        training=training)
        out = self.dropout2.forward(out,   training=training)
        if out is None:
            out = self.dropout2.output

        # skip-add + final ReLU
        skip = self.cache['x_skip']
        s   = out + skip
        m2  = (s > 0)
        out = s * m2
        self.cache['mask2'] = m2

        self.output = out
        return self.output

    def backward(self, d_out):
        """
        Backward pass through final ReLU, second branch, first ReLU, and skip addition.
        """
        # final ReLU
        m2    = self.cache['mask2']
        d_sum = d_out * m2

        # split: one copy through conv→bn→dropout, one straight to the skip
        d_res  = d_sum
        d_skip = d_sum

        # backward through second branch
        self.dropout2.backward(d_res)
        d_res = self.dropout2.dinputs

        d_res = self.bn2.backward(d_res)

        self.conv2.backward(d_res)
        d_res = self.conv2.dinputs

        # backward through ReLU #1
        m1    = self.cache['mask1']
        d_res = d_res * m1

        # backward through first branch
        self.dropout1.backward(d_res)
        d_res = self.dropout1.dinputs

        d_res = self.bn1.backward(d_res)

        self.conv1.backward(d_res)
        d_res = self.conv1.dinputs

        # combine with skip gradient
        dx = d_res + d_skip
        self.dinputs = dx
        return self.dinputs

class Layer_Reshape:
    """
    Simple reshape layer for flattening or plotting.
    """
    def __init__(self, shape):
        """Args:
            shape (tuple): Desired new shape excluding batch dimension.
        """
        self.shape = shape
        self.input_shape = None

    def forward(self, x, training=True):
        """
        Forward pass: reshape x to (batch_size, *shape).
        """
        self.input_shape = x.shape
        self.output =  x.reshape(x.shape[0], *self.shape)
        return self.output

    def backward(self, dvalue):
        """
        Backward pass: restore gradient shape to input_shape.
        """
        self.dinputs = dvalue.reshape(self.input_shape)
        return self.dinputs