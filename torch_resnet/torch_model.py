"""
resnet_torch.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Defines a small convolutional ResNet architecture with residual blocks, batch normalization,
             dropout, and adaptive pooling for CIFAR-10 classification.
Published: 04-26-2025
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic 3x3 convolutional residual block with identity skip connection.

    Consists of two Conv-BatchNorm-Dropout-ReLU sequences and an additive skip, followed by a ReLU.
    """
    def __init__(self, channels):
        """
        Initialize the residual block.

        Args:
            channels (int): Number of input and output feature channels.
        """
        super().__init__()
        # First convolutional sequence
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout1 = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional sequence
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Output tensor of same shape.
        """
        identity = x # Save identity for skip connection

        # First conv → BN → Dropout → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        # Second conv → BN → Dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Skip connection and final ReLU
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    A small ResNet for CIFAR-10 classification.

    Architecture:
        - Initial 3x3 conv, BN, ReLU
        - 4 stages each with a ResidualBlock and downsampling (via pool or conv+pool)
        - Global adaptive average pooling
        - Fully connected classifier
    """
    def __init__(self, num_classes=10):
        """
        Initialize the ResNet model.

        Args:
            num_classes (int): Number of classes for the final classification layer.
        """
        super().__init__()
        # Initial conv: 3->64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: Residual block + downsample (64 channels)
        self.layer1 = ResidualBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: Expand channels to 128, residual block, downsample
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = ResidualBlock(128)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: Expand channels to 256, residual block, downsample
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = ResidualBlock(256)
        self.pool3 = nn.MaxPool2d(2)

        # Stage 4: Expand channels to 512, res block, downsample
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = ResidualBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Replaces final pooling

        # Final classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the ResNet.

        Args:
            x (Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            Tensor: Logits of shape (N, num_classes).
        """
        # Initial layers
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)
        x = self.pool1(x)

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.pool2(x)

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.pool3(x)

        # Stage 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x