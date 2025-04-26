"""
train_numpy.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Training script for a custom ResNet-style convolutional neural network using a pure NumPy backend.
             It handles backend configuration, model building with configurable initialization and learning rate schedules,
             training with logging, and saving model parameters and architecture.
Published: 04-26-2025
"""

import os
import time
import tyro
from dataclasses import dataclass
# Model components and layers
from numpy_resnet.numpy_model import Model
from numpy_resnet.model_builder.layers import (Layer_Conv2D, Layer_BatchNorm2d, Layer_MaxPool2d,
                                   Layer_AdaptiveAvgPool2d, Layer_ResidualBlock, Layer_Reshape, Layer_Dense)
# Metrics, activations, optimizers, loss, and scheduling
from numpy_resnet.utils.metrics import Accuracy_Categorical
from numpy_resnet.model_builder.activation_functions import Activation_ReLU, Activation_Softmax
from numpy_resnet.model_builder.optimizers import Optimizer_Adamax
from numpy_resnet.model_builder.loss_functions import Loss_CategoricalCrossentropy
from numpy_resnet.model_builder.lr_schedules import ExponentialDecaySchedule, StepDecaySchedule
from numpy_resnet.utils import backend
# Utility functions for data loading, logging, and checkpoint directories
from common_utils.cifar10_data_loader import load_cifar10
from common_utils.logger import setup_logger
from common_utils.utils import get_new_run_dir


@dataclass
class CIFAR10NumpyTrainer:
    """
    Trainer configuration for CIFAR-10 using a NumPy-based ResNet model.

    Attributes:
        lr (float): Initial learning rate for the optimizer.
        final_lr (float): Final learning rate after decay.
        dropout_rate (float): Dropout rate applied in layers (if used).
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        schedule_type (str): Type of LR schedule: 'exponential' or 'step'.
        init_type (str): Weight initialization type (e.g., 'he_scaling').
        decay (float): L2 regularization coefficient.
        device (str): Compute backend to use: 'gpu' or 'cpu'.
    """
    #hyperparameters
    lr: float = 1e-3
    final_lr: float = 0.0001
    dropout_rate: float = 0.2
    epochs: int = 123
    batch_size: int = 128
    schedule_type: str = "exponential"
    init_type: str = "he_scaling"
    decay: float = 1e-4
    device: str = "gpu"

    def __post_init__(self):
        """
        Initialize backend, learning rate schedule, load CIFAR-10 data, and prepare logging.
        """
        # Configure compute backend (NumPy CPU or GPU via CuPy)
        backend.set_backend('gpu' if self.device == 'gpu' else 'cpu')

        # Set up learning rate decay schedule based on user choice
        if self.schedule_type == "exponential":
            self.schedule = ExponentialDecaySchedule(self.lr, self.final_lr, self.epochs)
        elif self.schedule_type == "step":
            self.schedule = StepDecaySchedule(self.lr, self.final_lr, self.epochs)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Load and optionally augment/normalize CIFAR-10 dataset
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = load_cifar10(augment=True, normalize=True)
        # Display dataset shapes for verification
        print("Training data shape:", self.x_train.shape)
        print("Training labels shape:", self.y_train.shape)
        print("Validation data shape:", self.x_val.shape)
        print("Validation labels shape:", self.y_val.shape)
        print("Test data shape:", self.x_test.shape)
        print("Test labels shape:", self.y_test.shape)

        # Create new checkpoint directory
        self.save_dir = get_new_run_dir(model_type="numpy")
        #set up logging
        self.logger = setup_logger(self.save_dir)

        # Initialize an empty ResNet model
        self.model = Model()
    
    def build_model(self):
        """
        Construct the ResNet architecture by stacking convolutional, normalization,
        activation, pooling, and residual blocks, followed by classifier layers.
        """
        # Initial convolutional block
        self.model.add(Layer_Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, 
                            init_type=self.init_type, weight_regularizer_l2=self.decay, use_im2col=True))
        self.model.add(Layer_BatchNorm2d(64))
        self.model.add(Activation_ReLU()) 

        # First residual stage
        self.model.add(Layer_ResidualBlock(64))
        self.model.add(Layer_MaxPool2d(kernel_size=2, stride=None))

        # Second stage: expand features to 128 channels
        self.model.add(Layer_Conv2D(64, 128, kernel_size=3, stride=1, padding=1, init_type=self.init_type,
                            weight_regularizer_l2=self.decay, use_im2col=True))
        self.model.add(Layer_BatchNorm2d(128))
        self.model.add(Activation_ReLU()) 
        self.model.add(Layer_ResidualBlock(128))
        self.model.add(Layer_MaxPool2d(kernel_size=2, stride=None))

        # Third stage: 256 channels
        self.model.add(Layer_Conv2D(128, 256, kernel_size=3, stride=1, padding=1, init_type=self.init_type,
                            weight_regularizer_l2=self.decay, use_im2col=True))
        self.model.add(Layer_BatchNorm2d(256))
        self.model.add(Activation_ReLU()) 
        self.model.add(Layer_ResidualBlock(256))
        self.model.add(Layer_MaxPool2d(kernel_size=2, stride=None))

        # Fourth stage: 512 channels
        self.model.add(Layer_Conv2D(256, 512, kernel_size=3, stride=1, padding=1, init_type=self.init_type,
                            weight_regularizer_l2=self.decay, use_im2col=True))
        self.model.add(Layer_BatchNorm2d(512))
        self.model.add(Activation_ReLU()) 
        self.model.add(Layer_ResidualBlock(512))

        # Global average pooling and classification head
        self.model.add(Layer_AdaptiveAvgPool2d((1, 1)))
        # 1) flatten 512×1×1 → 512
        self.model.add(Layer_Reshape((512,)))
        # 2) dense layer to 10 classes
        self.model.add(Layer_Dense(512, 10, init_type=self.init_type, weight_regularizer_l2=self.decay))
        self.model.add(Activation_Softmax()) # output probability distribution

        # Attach loss, optimizer, and metric
        self.model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adamax(self.lr, self.schedule), accuracy=Accuracy_Categorical())

        # Finalize the model
        self.model.finalize()
    
    def train(self):
        """
        Train the model on the training set with validation, logging time and progress.
        """
        start_time = time.time()
        self.logger.info("Starting training...")
        # Kick off the training loop: handles batching, forward/backward passes, and metrics
        self.model.train(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size, 
            print_every=1, augment_data=True)
        total_time = time.time() - start_time
        self.logger.info(f"\n Total training time: {total_time:.2f} seconds on {self.device}")

    def save(self):
        """
        Save trained model parameters and full model architecture to disk.
        """
        # Retrieve model parameters
        parameters = self.model.get_parameters() # extract number of layers
        self.logger.info(f"Saved model parameters with {len(parameters)} layers.")
        # save parameters and model
        self.model.save_parameters(os.path.join(self.save_dir, 'cifar_10.params'))
        self.model.save(os.path.join(self.save_dir, 'cifar_10.model'))

    def run(self):
        """
        Orchestrate full workflow: build the network, train, then save results.
        """
        self.build_model()
        self.train()
        self.save()

if __name__ == "__main__":
    # Parse CLI args and initiate training
    trainer: CIFAR10NumpyTrainer = tyro.cli(CIFAR10NumpyTrainer)
    trainer.run()