"""
cifar10_numpy_inference.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: Provides a CLI tool for loading a trained NumPy-based ResNet model,
             evaluating it on the CIFAR-10 validation set, and performing inference
             on one or more external images with top-3 class prediction output.
Published: 04-26-2025
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
import tyro
from numpy_resnet.numpy_model import Model
from common_utils.cifar10_data_loader import load_cifar10

@dataclass
class CIFAR10NumpyInference:
    """
    CLI inference helper for NumPy-based CIFAR-10 models.

    Attributes:
        checkpoint_path (str): Path to saved model checkpoint (.model or .params file).
        image_path (str): Path to an image file or directory of images for inference.
    """
    checkpoint_path: str = ""
    image_path: str = ""

    def __post_init__(self):
        """
        Load data, model checkpoint, and prepare image list.
        Evaluates model on validation set for quick sanity check.
        """

        # Load CIFAR-10 data (with augmentation/normalization flags ignored here)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(augment=True, normalize=True)

        # set up CIFAR-10 class labels (for human-readable outputs)
        class_labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",}

        # Build list of images to process for inference
        if os.path.isfile(self.image_path):
            self.image_list = [self.image_path]
        elif os.path.isdir(self.image_path):
            # Collect image files with common extensions
            self.image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)
                               if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            if len(self.image_list) == 0:
                raise ValueError(f"No images found in directory: {self.image_path}")
        else:
            raise ValueError(f"Provided path {self.image_path} is neither a file nor a directory.")
        self.image_list = sorted(self.image_list)

        # Load model from checkpoint
        self.model = Model.load(self.checkpoint_path)

        # Evaluate on validation set to report baseline accuracy
        self.model.evaluate(x_val, y_val)


    def predict_single(self, image_path):
        """
        Preprocess a single image, run model prediction, and return top-3 classes.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List[Tuple[int, str, float]]: Top-3 (class_idx, class_name, confidence%) tuples.
        """
        # Read image in BGR then convert to RGB
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        # Resize to CIFAR-10 input size
        image_data = cv2.resize(image_data, (32, 32))
        # Flatten and normalize to [0,1]
        image_data = (image_data.reshape(1, -1).astype(np.float32)) / 255.0

        # Forward pass: raw logits or scores
        confidences = self.model.predict(image_data)  # shape (1, 10)

        # Apply softmax for probabilities
        exp_values = np.exp(confidences - np.max(confidences, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Identify top-3 predictions
        top3_indices = np.argsort(probabilities[0])[::-1][:3]  # sort and take top 3
        top3_classes = []
        for idx in top3_indices:
            class_idx = idx
            class_name = self.class_labels[class_idx]
            confidence_score = probabilities[0][idx] * 100
            top3_classes.append((class_idx, class_name, confidence_score))

        return top3_classes
        
    def predict(self):
        """
        Run inference on all specified images and print results.

        Returns:
            If single image: top-3 list; otherwise list of (path, top-3 list) tuples.
        """
        # Single-image mode
        if len(self.image_list) == 1:
            """Predict on a single image."""
            top3 = self.predict_single(self.image_list[0])
            print(f"Top-3 predictions for {self.image_list[0]}:")
            for class_idx, class_name, confidence in top3:
                print(f"    {class_idx} ({class_name}) with {confidence:.2f}% confidence")
            print()  # add empty line between images
            return top3
        
        else:
            # Batch mode for multiple images
            all_preds = []
            for path in self.image_list:
                top3 = self.predict_single(path)
                all_preds.append((path, top3))

            print("Predicted classes:")
            for path, top3 in all_preds:
                print(f"Top-3 predictions for {path}:")
                for class_idx, class_name, confidence in top3:
                    print(f"    {class_idx} ({class_name}) with {confidence:.2f}% confidence")
                print()  # add empty line between images
            return all_preds


if __name__ == "__main__":
    # Parse CLI arguments and perform inference
    inferencer: CIFAR10NumpyInference = tyro.cli(CIFAR10NumpyInference)
    inferencer.predict()