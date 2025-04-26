"""
cifar10_pytorch_inference.py

Project: Torch-NumPy-CIFAR-CNN
Author: Gabriel Souza
Description: CLI tool for loading a trained PyTorch ResNet model, processing CIFAR-10-style images
             (single file or directory), and outputting top-3 class predictions with confidence scores.
Published: 04-26-2025
"""

import os
import tyro
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch_resnet.torch_model import ResNet
import torchvision.transforms as transforms
from PIL import Image

@dataclass
class CIFAR10PytorchInference:
    """
    Command-line inference helper for PyTorch ResNet on CIFAR-10 images.

    Attributes:
        checkpoint_path (str): Path to saved PyTorch model (.pth) checkpoint.
        image_path (str): Path to input image file or directory of images.
        device (str): Preferred device string, e.g., 'cuda' or 'cpu'.
    """

    checkpoint_path: str = ""
    image_path: str = ""
    device: str = "cuda"

    def __post_init__(self):
        """
        Initialize inference: load class labels, collect image files,
        set device, load model weights, and prepare preprocessing.
        """
        # set up CIFAR-10 class labels (for human-readable outputs)
        self.class_labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",}

        # Build list of image paths for inference
        if os.path.isfile(self.image_path):
            self.image_list = [self.image_path]
        elif os.path.isdir(self.image_path):
            # Filter image files by common image file extensions
            self.image_list = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)
                               if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            if len(self.image_list) == 0:
                raise ValueError(f"No images found in directory: {self.image_path}")
        else:
            raise ValueError(f"Provided path {self.image_path} is neither a file nor a directory.")
        self.image_list = sorted(self.image_list)

        # Select compute device
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # Load and prepare the ResNet model
        if self.checkpoint_path is None:
            raise ValueError("You must provide a checkpoint_path to load the trained model.")
        self.model = ResNet()
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing pipeline: resize, tensorize, normalize
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),# CIFAR10 images are 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # adjust if your train normalization is different
        ])

    def predict_single(self, image_path):
        """
        Load and preprocess an image, run through the model, and return top-3 predictions.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List[Tuple[int, str, float]]: Each tuple is (class_idx, class_name, confidence_percent).
        """
        # Load image and convert to RGB
        img = Image.open(image_path).convert("RGB")
        # Apply preprocessing transforms
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            probs = F.softmax(outputs, dim=1) # Use softmax to get probability score
            # Get top probability and index of top 3 classes
            topk_confidences, topk_preds = torch.topk(probs, k=3, dim=1)
        
        # Map to labels and percentage
        topk_classes = []
        for idx, conf in zip(topk_preds[0], topk_confidences[0]):
            class_idx = idx.item()
            class_name = self.class_labels[class_idx]
            confidence_score = conf.item() * 100
            topk_classes.append((class_idx, class_name, confidence_score))

        return topk_classes

    def predict(self):
        """
        Run inference on all images and print formatted results.

        Returns:
            If a single image: List of top-3 predictions;
            otherwise List of (path, top-3 list) tuples.
        """
        if len(self.image_list) == 1:
            """Predict on a single image."""
            topk = self.predict_single(self.image_list[0])
            print(f"Top-3 predictions for {self.image_list[0]}:")
            for class_idx, class_name, confidence in topk:
                print(f"    {class_idx} ({class_name}) with {confidence:.2f}% confidence")
            print()  # add empty line between images
            return topk
            
        else:
            # Batch mode
            all_preds = []
            for path in self.image_list:
                topk = self.predict_single(path)
                all_preds.append((path, topk))

            print("Predicted classes:")
            for path, topk in all_preds:
                print(f"Top-3 predictions for {path}:")
                for class_idx, class_name, confidence in topk:
                    print(f"    {class_idx} ({class_name}) with {confidence:.2f}% confidence")
                print()  # add empty line between images
            return all_preds
    
if __name__ == "__main__":
    # Parse CLI arguments and perform inference
    inferencer: CIFAR10PytorchInference = tyro.cli(CIFAR10PytorchInference)
    inferencer.predict()