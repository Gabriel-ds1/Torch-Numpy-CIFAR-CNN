# ⚡ Torch-Numpy-CIFAR-CNN

<p align="center"> <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"> <img src="https://img.shields.io/badge/frameworks-NumPy%20%7C%20PyTorch-blue.svg" alt="Frameworks"> <img src="https://img.shields.io/badge/logging-TensorBoard%20%7C%20W%26B-orange.svg" alt="Logging"> <img src="https://img.shields.io/badge/cupy-support-lightgrey.svg" alt="CuPy Support"> </p>

🚀 A dual-framework Convolutional Neural Network (CNN) project for CIFAR-10 — implemented both from scratch using pure NumPy (and optional CuPy) and using PyTorch.
Demonstrates deep understanding of convolution operations, backpropagation through Conv layers, skip connections, and optimization trade-offs between custom and high-performance libraries.

---

## 📚 Table of Contents

- [Features](#-features)
- [Highlights](#-highlights)
- [GUI Demo](#-gui-demo-tkinter)
- [Performance Notes](#-performance-notes)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Why This Project](#-why-this-project)
- [License](#-license)
- [Contact](#-contact)

---

## 📝 Features

- _Full CNN with ResNet-style skip connections_ in both _pure NumPy_ and _PyTorch_:
  -- 4 residual stages (initial conv + 4 blocks) using 3×3 convolutions, identity shortcuts, and ReLU activations.

- _Custom NumPy implementations of_:
  -- `Conv2D`, `BatchNorm2D`, `MaxPool2D`, `AdaptiveAvgPool2D`
  -- ResNet block with Skip Connections
  -- Activation functions (`ReLU`, `LeakyReLU`, `Softmax`, `Linear`)
  -- Optimizers (`SGD` (with momentum), `Adagrad`, `RMSprop`, `Adam`, `Adamax`)
  -- Learning Rate Schedulers: Exponential & Step decay
  -- Loss Functions (`Cross-Entropy`, `Binary Cross-Entropy`, `MSE`, `MAE`)
  -- im2col optimization for faster NumPy-CNN training

- _CuPy_ support: Run NumPy models on GPU (though note: significantly slower than PyTorch due to lack of fused ops and other optimizations).

- _Gradient Checking_: `gradient_check.py` compares the backward pass of the custom Conv2D against PyTorch's torch.nn.Conv2d, verifying correct backpropagation.

- _Inference Utilities_:
  -- `inference_numpy.py`: CLI tool for NumPy model inference on images (top-3 predictions).
  -- `inference_pytorch.py`: Same for PyTorch model using ResNet class.

- _Logging & Visualization (PyTorch)_:
  -- TensorBoard (`tensorboard`) and Weights & Biases (`wandb`) logging for PyTorch training.

- _CLI Interface_: Powered by _Tyro_ for easy argument parsing.

---

## ⭐ Highlights

✅ Pure NumPy CNN with manual backpropagation through convolution, pooling, batch norm, etc.
✅ CuPy optional backend for NumPy model (experimental).
✅ Gradient checking ensures mathematical correctness of convolution gradients.
✅ Modular design: clean separation of models, optimizers, losses, and utilities.
✅ Full PyTorch version to compare efficiency and optimization.
✅ TensorBoard & W&B logging integrated out-of-the-box.
✅ Configurable training parameters with Tyro CLI parsing (learning rate, optimizer, batch size, epochs, backend, etc.).
✅ Modular, Well-Documented Codebase: thorough docstrings, comments, and a clean folder structure.

---

## 🖥️ GUI Demo (Tkinter)

![GUI CNN Demo](https://i.imgur.com/Y9tRzaw.gif)

Using the included `gui.py`, you can load any trained model and view the output classification:

```bash
python gui.py --checkpoint_path model_ckpt/pytorch/001/resnet_cifar10_best.pth
```

-Upload a folder with any image containing: {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", or "truck"}

-See the predicted class and probability bar chart.

---

## 🔥 Performance Notes

(NumPy results limited by training speed even with GPU.)

| Framework | Val Accuracy   | Train Accuracy |
| --------- | -------------- | -------------- |
| PyTorch   | 93.74%         | 99.24%         |
| NumPy     | (not full-run) | (not full-run) |

- _PyTorch Implementation_: Trained to _93.74%_ validation accuracy (99.24% train).
- _NumPy Implementation_:
  -- Optional `--device_str gpu` uses CuPy for GPU acceleration, but current pure-Python loops remain significantly slower than PyTorch’s optimized kernels.
  -- Suitable for educational comparison and debugging, but not recommended for full-scale training due to performance constraints.

![wandb charts](https://i.imgur.com/4zAs45f.png)

---

## 🗂️ Project Structure

```bash
├─ common_utils/                          # Utility modules
│   ├── cifar10_data_loader.py            # download, load, augment CIFAR-10 (NumPy/CuPy)
│   ├── logger.py                         # console + file logging setup
│   ├── utils.py                          # run-dir creation, misc helpers
│
├─ model_ckpt/                            # Checkpoint directories
│   ├── numpy/                            # NumPy training runs
│   ├── pytorch/                          # PyTorch training runs
│
├─ numpy_resnet/                          # Pure NumPy/CuPy model builder
│   ├── model_builder/                    # Layer, activations, losses, optimizers, schedules
│   │     ├── activation_functions.py     # ReLU, LeakyReLU, Softmax (NumPy)
│   │     ├── layers.py                   # Conv2D, BatchNorm2D, MaxPool2D, etc.
│   │     ├── loss_functions.py           # CrossEntropyLoss (NumPy)
│   │     ├── lr_schedules.py             # Learning rate schedulers
│   │     ├── optimizers.py               # SGD, Adam (NumPy)
│
│   ├── utils/                            # Backend switching, metrics, im2col vectorization
│   │     ├── backend.py                  # NumPy vs. CuPy backend switching
│   │     ├── metrics.py                  # Accuracy, precision, recall calculators
│   │     ├── vectorize.py                # Utility functions for vectorized operations
│
│   ├── numpy_model.py                    # Orchestrates main NumPy ResNet model -> build, train, save
│
├─ test_images/                           # Example images for inference
│   ├── bird.jpg
│   ├── cat.jpg
│   ├── dog.jpg
│   ├── truck.jpg
│
├─ torch_resnet/                           # PyTorch ResNet implementation
│   ├── utils/
│   │     ├── data_loader_torch.py        # PyTorch DataLoader for CIFAR-10
│   ├── torch_model.py                    # Main PyTorch ResNet model
│
gradient_check.py       # Compares custom Conv2D gradients to PyTorch Conv2D
inference_numpy.py      # Run inference with NumPy model
inference_pytorch.py    # Run inference with PyTorch model
train_numpy.py          # Pure NumPy training script
train_pytorch.py        # PyTorch training script
README.md
```

Each script is documented with docstrings and comments to guide understanding and modification.

---

## 🚀 Getting Started

### Model Download

You can download the pre-trained PyTorch model and run inference:

📥 Pre-trained PyTorch model:
[Numpy-Torch-CIFAR-CNN-Weights](https://drive.google.com/uc?export=download&id=130HBzNBcYNox2Ean7sIeuQMdEKNUHk4G)

Save the downloaded model inside `model_ckpt/pytorch/001/`.

### 1. Clone the Repository

```bash
git clone https://github.com/Gabriel-ds1/Torch-Numpy-CIFAR-CNN.git
cd Torch-Numpy-CIFAR-CNN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Requires:

-NumPy

-CuPy

-PyTorch (v2.7.0)

-Tyro

-OpenCV

-Weights & Biases (wandb)

-TensorBoard

-Pickle

-SciPy

-Pillow

-Tkinter

### 3. Run Training

_Pytorch Training_

```bash
python train_pytorch.py --lr 0.001 --batch_size 128 --epochs 200 --vis_logging wandb
```

_NumPy Training (Experimental, Slow):_

```bash
python train_numpy.py --lr 0.001 --batch_size 128 --epochs 50 --schedule_type exponential --device gpu
```

### 4. Run Inference

_PyTorch Inference_

```bash
python inference_pytorch.py --checkpoint_path model_ckpt/pytorch/001/resnet_cifar10_best.pth --image_path test_images/
```

_NumPy Inference_

```bash
python inference_numpy.py --checkpoint_path model_ckpt/numpy/001/cifar10.model --image_path test_images/
```

---

## ✨ Why This Project?

I took on this side project to fully understand the math and underlying ideas of:

    - How convolution really works (stride, padding, dilation)
    - How backpropagation through Conv2D is computed manually
    - Why libraries like PyTorch are highly optimized (fused ops, kernels)
    - How to debug model architectures and gradient flows
    - Understanding every detail inside a deep learning model

_Final Note_

✅ Every line of the custom NumPy implementation was built from scratch.
✅ No hidden frameworks were used inside the NumPy ResNet.
✅ Gradient correctness verified via gradient_check.py.
✅ Full training pipeline, logging, CLI parsing, checkpoint saving — production-style engineering practices.

---

## 📜 License

This project is released under the MIT License. Feel free to use, modify, and distribute.

---

## 📣 Contact

If you're a recruiter or collaborator interested in AI, deep learning, or research-driven work — feel free to connect!

> 📧 Email: [gabesouza004@gmail.com](mailto:gabesouza004@gmail.com)
