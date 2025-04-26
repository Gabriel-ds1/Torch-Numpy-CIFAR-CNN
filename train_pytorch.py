""""
File: train_pytorch.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Training script for a ResNet-based CNN on the CIFAR-10 dataset using PyTorch. 
             This script handles data loading, model setup, training with optional early stopping 
             and learning rate scheduling, logging to TensorBoard or Weights & Biases, and final evaluation.
Published: 04-26-2025
"""

import os
import time
import numpy as np
import tyro
import random
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_resnet.utils.data_loader_torch import CIFARDataset, unnormalize
from torch_resnet.torch_model import ResNet
from common_utils.logger import setup_logger
from common_utils.cifar10_data_loader import load_cifar10
from common_utils.utils import get_new_run_dir
from dataclasses import dataclass

@dataclass
class CIFAR10Pytorch:
    """
    Configuration and training handler for CIFAR-10 classification using a ResNet model in PyTorch.

    Attributes:
        lr (float): Initial learning rate for the optimizer.
        batch_size (int): Number of samples per training batch.
        epochs (int): Maximum number of training epochs.
        early_stopping (int): Stop training if no improvement after this many epochs.
        weight_decay (float): L2 regularization factor for optimizer.
        scheduler_patience (int): Epochs to wait before reducing LR on plateau.
        scheduler_factor (float): Factor by which to reduce the learning rate.
        vis_logging (str): Logging framework to use: 'wandb', 'tensorboard', or None.
    """
    # Parameters
    lr: float = 1e-3
    batch_size: int = 128
    epochs:int = 500
    early_stopping: int = 20 # stop if no improvement in validation for these epochs
    weight_decay: float = 1e-4
    # Reduce LR on Plateau Params
    scheduler_patience: int = 10 # wait epochs before LR reduction
    scheduler_factor: float = 0.5 # LR reduction factor
    # visualizer logging
    vis_logging: str = "wandb" # 'wandb', 'tensorboard', or None

    def __post_init__(self):
        """
        Initialize reproducibility, device, logging directories, and visualization setups.
        """
        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Select device: GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directory for saving checkpoints and logs
        self.save_dir = get_new_run_dir(model_type="pytorch")

        # Set up logging and visualization
        # tensorboard --logdir=runs (for tensorboard)
        # https://wandb.ai/ (for wandb)
        if self.vis_logging == "tensorboard":
            # TensorBoard writer
            self.writer = SummaryWriter(log_dir=self.save_dir)
        elif self.vis_logging == "wandb":
            # Initialize Weights & Biases run
            wandb.init(project="cifar10", config={"lr": self.lr, "batch_size": self.batch_size})
        # Logger for console/file output
        self.logger = setup_logger(self.save_dir)

    def data_preprocess(self):
        """
        Load CIFAR-10 dataset, create PyTorch datasets and data loaders.
        """
        # Fetch pre-split CIFAR-10 arrays
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = load_cifar10()
        # Display dataset shapes for verification
        print("Training data shape:", self.x_train.shape)
        print("Training labels shape:", self.y_train.shape)
        print("Validation data shape:", self.x_val.shape)
        print("Validation labels shape:", self.y_val.shape)
        print("Test data shape:", self.x_test.shape)
        print("Test labels shape:", self.y_test.shape)

        # Wrap arrays in PyTorch Dataset objects, apply augmentation to training data
        self.train_dataset = CIFARDataset(self.x_train, self.y_train, augment=True)
        self.val_dataset = CIFARDataset(self.x_val, self.y_val, augment=False)  # No transform
        self.test_dataset = CIFARDataset(self.x_test, self.y_test, augment=False)  # No transform
        # Create data loaders for batching
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

    def setup_model(self):
        """
        Instantiate the ResNet model, loss criterion, optimizer, and learning rate scheduler.
        """
        print("device:", self.device)
        # Initialize ResNet and move to device
        self.model = ResNet().to(self.device)

        # Cross-entropy loss for classification
        self.criterion = nn.CrossEntropyLoss()

        # Adamax optimizer with weight decay
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.scheduler_patience, factor=self.scheduler_factor)

    def train(self):
        """
        Execute the full training loop: preprocessing, forward/backward passes, logging,
        early stopping, checkpointing, and final testing.
        """
        # Prepare data and model
        self.data_preprocess()
        self.setup_model()
        
        # Setup visualization: watch model for gradients in wandb or sample images in TensorBoard
        if self.vis_logging == "tensorboard":
            # single batch of unnormalized input images for tensorboard visualization
            sample_batch_tensor, _ = next(iter(self.train_loader))  # (B, C, H, W)
        elif self.vis_logging == "wandb":
            wandb.watch(self.model, self.criterion, log="all")

        self.start_time = time.time()
        self.logger.info("Starting training...")

        best_val_acc = 0.0
        epochs_no_improve = 0

        # Main epoch loop
        for epoch in range(1, self.epochs+1):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            val_loss_total = 0.0

            # Training batches
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_train += preds.eq(target).sum().item()

            # Compute training accuracy
            train_acc = correct_train / len(self.train_loader.dataset)

            # Validation phase
            self.model.eval()
            correct_val = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, target)
                    val_loss_total += loss.item() * data.size(0)  # Total loss per batch
                    preds = outputs.argmax(dim=1)
                    correct_val += preds.eq(target).sum().item()

            # Average metrics
            avg_train_loss = running_loss / len(self.train_loader)
            val_loss = val_loss_total / len(self.val_loader.dataset)
            val_acc = correct_val / len(self.val_loader.dataset)

            # Adjust learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Logging
            if self.vis_logging == "tensorboard":
                # # Log scalars
                self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)
                # Log images
                sample_batch_tensor, _ = next(iter(self.train_loader))
                unnormed = unnormalize(sample_batch_tensor[:16])
                self.writer.add_images("Sample images", unnormed, epoch)

            elif self.vis_logging == "wandb":
                # Log to Weights & Biases (wandb)
                wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "learning_rate": self.scheduler.get_last_lr()[0]
                })

            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save checkpoint
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"resnet_cifar10_best.pth"))
            else:
                epochs_no_improve += 1

            # Early stopping condition
            if epochs_no_improve >= self.early_stopping:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Console output for current epoch
            print(f"Epoch {epoch}, Loss: {running_loss/len(self.train_loader):.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%, LR: {self.scheduler.get_last_lr()[0]:.6f}")

        # Evaluate on test set after training completes
        self.test()

    def test(self):
        """
        Evaluate the trained model on the held-out test dataset and log the final accuracy and elapsed time.
        """
        correct_test = 0
        self.model.eval()

        # Compute predictions on test set
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                preds = outputs.argmax(dim=1)
                correct_test += preds.eq(target).sum().item()
        # Final test accuracy
        test_acc = correct_test / len(self.test_loader.dataset)
        print(f"Final Test Accuracy: {test_acc*100:.2f}%")

        # Log total runtime
        total_time = time.time() - self.start_time
        self.logger.info(f"\n Total training time: {total_time:.2f} seconds on {self.device}")

if __name__ == "__main__":
    # Parse CLI arguments and start training
    trainer: CIFAR10Pytorch = tyro.cli(CIFAR10Pytorch)
    trainer.train()