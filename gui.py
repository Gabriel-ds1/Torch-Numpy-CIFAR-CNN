"""
gui.py

Project: Torch-Numpy-CIFAR-CNN
Author: Gabriel Souza
Description: Tkinter-based GUI for uploading CIFAR-10-style images (single or batch) 
             and running inference using a trained PyTorch ResNet model.
Published: 2025-04-26
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F
from torch_resnet.torch_model import ResNet
import torchvision.transforms as transforms
import os

class CIFAR10GUIApp:
    """
    GUI application for uploading and classifying CIFAR-10 images using a trained ResNet model.
    """
    def __init__(self, root, checkpoint_path, device="cuda"):
        self.root = root
        self.root.configure(bg="black")
        self.root.title("Torch-Numpy-CIFAR-CNN - CIFAR10 Classifier")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ResNet()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Class labels
        self.class_labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
                             5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

        # Transform
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize_with_padding),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # UI Components
        self.image_label = tk.Label(root, text="No Image Loaded", font=("Helvetica", 14),
                                    bg = "#555555",
                                    fg = "white",     # text color
                                    bd=2            # thin border
                                    )
        self.image_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=256, height=256, bd=2, relief="ridge", 
                                    bg="#555555",   # light gray background
                                    )
        self.canvas.pack(pady=10)

        tk.Button(root, text="Upload Folder", bg="#555555", fg="white", command=self.upload_folder).pack(pady=5)
        tk.Button(root, text="Clear", bg="#555555", fg="white", command=self.clear_canvas).pack(pady=5)

        nav_frame = tk.Frame(root, bg="#555555")
        nav_frame.pack(pady=5)

        tk.Button(nav_frame, text="<< Prev", bg="#555555", fg="white", command=self.previous_image).grid(row=0, column=0, padx=10)
        tk.Button(nav_frame, text="Next >>", bg="#555555", fg="white", command=self.next_image).grid(row=0, column=1, padx=10)

        self.result_text = tk.Text(root, height=4, width=50, font=("Helvetica", 12), 
                                   bg="#555555",   # light gray background
                                    fg="white",     # text color
                                    relief="ridge",
                                    bd=2            # thin border
                                    )
        
        self.result_text.pack(pady=10)

        # Bar chart canvas for probability visualization
        self.BAR_WIDTH = 400
        self.BAR_HEIGHT = 300
        self.CANVAS_WIDTH = 460
        self.CANVAS_HEIGHT = 300
        self.prob_canvas = tk.Canvas(root, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="#555555", bd=2, relief='ridge')
        self.prob_canvas.pack(pady=10)
        self.result_text.tag_configure("center", justify='center')

        self.loaded_images = []  # List of image file paths (for folders)
        self.current_idx = 0     # Track which image is being shown if folder

    def upload_folder(self):
        """Load a directory of images for classification."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.loaded_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
            if not self.loaded_images:
                messagebox.showerror("Error", "No valid images found in selected folder.")
                return
            self.current_idx = 0
            self.show_and_predict(self.loaded_images[self.current_idx])

    def next_image(self):
        """Go to the next image if available."""
        if self.loaded_images and self.current_idx < len(self.loaded_images) - 1:
            self.current_idx += 1
            self.show_and_predict(self.loaded_images[self.current_idx])

    def previous_image(self):
        """Go to the previous image if available."""
        if self.loaded_images and self.current_idx > 0:
            self.current_idx -= 1
            self.show_and_predict(self.loaded_images[self.current_idx])

    def show_and_predict(self, img_path):
        """Display the image and run prediction."""
        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((256, 256))
            img_tk = ImageTk.PhotoImage(img_resized)
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.canvas.image = img_tk  # prevent garbage collection
            self.image_label.config(text=os.path.basename(img_path))

            # Preprocess and predict
            transformed_img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(transformed_img)
                probs = F.softmax(outputs, dim=1)
                self.draw_prob_bar_chart(probs)
                topk_confidences, topk_preds = torch.topk(probs, k=3, dim=1)

            results = []
            for idx, conf in zip(topk_preds[0], topk_confidences[0]):
                label = self.class_labels[idx.item()]
                confidence = conf.item() * 100
                results.append(f"{label} ({confidence:.2f}%)")

            # Update result text
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Top-3 Predictions:\n", "center")
            for line in results:
                self.result_text.insert(tk.END, f"{line}\n", "center")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw_prob_bar_chart(self, probs):
        """
        Draws a horizontal bar chart of output probabilities.

        Args:
            probs (torch.Tensor): Output probabilities from the model.
        """
        # Clear previous bars
        self.prob_canvas.delete('all')
        probs = probs.flatten().cpu().numpy()
        n = len(probs)
        bar_h = int(self.BAR_HEIGHT / n) - 4  # Bar height per class

        max_prob = np.max(probs)

        for i, (prob, label) in enumerate(zip(probs, self.class_labels.values())):
            bar_length = int(prob * (self.BAR_WIDTH - 120))

            y0 = i * (bar_h + 4) + 6
            y1 = y0 + bar_h

            color = '#448aff' if prob < max_prob else '#43a047'  # Highlight highest prob class

            # Draw bar
            self.prob_canvas.create_rectangle(110, y0, 110 + bar_length, y1, fill=color, outline='black', width=1)
            # Draw label
            self.prob_canvas.create_text(10, y0 + bar_h // 2, anchor='w',
                                         text=str(label), font=("Helvetica", 12), fill="white")
            # Draw probability %
            self.prob_canvas.create_text(
                110 + bar_length + 8, y0 + bar_h // 2, anchor='w',
                text=f"{prob:.2%}", font=("Helvetica", 10), fill="white"
            )

    def resize_with_padding(self, img, target_size=(32,32)):
        """
        Resize an image while keeping aspect ratio, then pad to target_size.

        Args:
            img (PIL.Image): Input image.
            target_size (tuple): Desired output size (width, height).

        Returns:
            PIL.Image: Transformed image.
        """
        original_width, original_height = img.size
        target_width, target_height = target_size

        # Calculate resize ratio
        ratio = min(target_width/original_width, target_height/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Resize maintaining aspect ratio
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create new blank canvas and paste resized image onto it (centered)
        new_img = Image.new("RGB", target_size, (0,0,0))  # Black padding
        upper_left_x = (target_width - new_width) // 2
        upper_left_y = (target_height - new_height) // 2
        new_img.paste(img, (upper_left_x, upper_left_y))

        return new_img

    def clear_canvas(self):
        """Clear canvas and result text."""
        self.canvas.delete("all")
        self.image_label.config(text="No Image Loaded")
        self.result_text.delete(1.0, tk.END)
        self.loaded_images = []
        self.current_idx = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained PyTorch .pth model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu)')
    args = parser.parse_args()

    root = tk.Tk()
    app = CIFAR10GUIApp(root, checkpoint_path=args.checkpoint_path, device=args.device)
    root.mainloop()