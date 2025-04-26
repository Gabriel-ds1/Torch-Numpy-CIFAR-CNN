"""
lr_schedules.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Provides learning rate scheduling strategies (Exponential and Step decay)
             for dynamic adjustment of optimizer learning rates across training epochs.
Published: 04-26-2025
"""

class ExponentialDecaySchedule:
    """
    Returns the learning rate for a given epoch based on an exponential decay schedule.
    
    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
    
    Returns:
        float: The learning rate for the current epoch.
    """
    def __init__(self, initial_lr, final_lr, total_epochs):
        """
        Initialize exponential decay scheduler parameters.

        Args:
            initial_lr (float): Learning rate at epoch 0.
            final_lr (float): Desired learning rate after total_epochs.
            total_epochs (int): Total epochs to evenly decay over.
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs

    def __call__(self, epoch):
        """
        Compute learning rate for a given epoch.

        Uses formula: lr = initial_lr * (decay_rate ** epoch), where
        decay_rate = (final_lr / initial_lr) ** (1 / total_epochs).

        Args:
            epoch (int): Current epoch index (0-based).

        Returns:
            float: Learning rate for this epoch.
        """
        # Determine per-epoch decay multiplier
        decay_rate = (self.final_lr / self.initial_lr) ** (1 / self.total_epochs)
        # Apply exponential decay
        return self.initial_lr * (decay_rate ** epoch)

class StepDecaySchedule:
    """
    Returns the learning rate for a given epoch based on a step decay schedule.

    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
        step_size (int): Number of epochs between each decay.

    Returns:
        float: The learning rate for the current epoch.
    """
    def __init__(self, initial_lr, final_lr, total_epochs, step_size=30):
        """
        Initialize step decay scheduler parameters.

        Args:
            initial_lr (float): Learning rate at epoch 0.
            final_lr (float): Desired learning rate after total_epochs.
            total_epochs (int): Total number of epochs in training.
            step_size (int): Epochs between each decay step.
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.step_size = step_size

    def __call__(self, epoch):
        """
        Compute learning rate for a given epoch using step decay.

        The total number of decay steps is total_epochs // step_size,
        and each step multiplies the previous LR by a fixed decay factor:

            decay_factor = (final_lr / initial_lr) ** (1 / num_steps)
            lr = initial_lr * (decay_factor ** num_steps_completed)

        Args:
            epoch (int): Current epoch index (0-based).

        Returns:
            float: Learning rate for this epoch.
        """
        # Calculate the number of decay events planned
        num_decays = self.total_epochs // self.step_size
        # Compute uniform decay factor per event
        decay_factor = (self.final_lr / self.initial_lr) ** (1 / max(1, num_decays))
        # Determine how many steps have occurred so far
        num_steps = epoch // self.step_size
        # Apply step decay
        return self.initial_lr * (decay_factor ** num_steps)