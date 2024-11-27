from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

from spottheplace.ml.utils import AddMask


class GradCam:
    def __init__(self, model_path: str) -> None:
        """
        Initialize the GradCam object.

        Args:
            - model_path: Path to the model weights file.
            - target_layer: The target layer to hook for Grad-CAM.
        """
        self.model = self._load_model(model_path)
        self.target_layer = self.model.layer4[2].conv3  # Last layer of the last block of ResNet50
        self.activations: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []

    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load the model and modify the fully connected layer to match the number of classes.

        Args:
            - model_path: Path to the model weights file.

        Returns:
            - The loaded and modified model.
        """
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
        model.eval()
        return model

    def _load_and_preprocess_image(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        """
        Load and preprocess the input image for the model.

        Args:
            - image_path: Path to the input image file.

        Returns:
            - A tuple containing the original image and the preprocessed image tensor.
        """
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            AddMask(),  # Add a mask to hide location information
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        return image, input_tensor

    def _register_hooks(self, layer: nn.Module) -> None:
        """
        Register forward and backward hooks to capture activations and gradients.

        Args:
            - layer: The target layer to hook.
        """
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        self.handle_forward = layer.register_forward_hook(forward_hook)
        self.handle_backward = layer.register_full_backward_hook(backward_hook)

    def _generate_heatmap(self, image: Image.Image, activations: List[torch.Tensor], gradients: List[torch.Tensor]) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap from activations and gradients.

        Args:
            - image: The original input image.
            - activations: The captured activations from the forward pass.
            - gradients: The captured gradients from the backward pass.

        Returns:
            - The generated heatmap as a numpy array.
        """
        grads = gradients[0].mean(dim=[2, 3], keepdim=True)
        cams = torch.relu((grads * activations[0]).sum(dim=1)).squeeze(0)
        cams = (cams - cams.min()) / (cams.max() - cams.min())
        cams = cams.detach().numpy()
        heatmap = np.uint8(255 * cams)
        heatmap = Image.fromarray(heatmap).resize(image.size)
        return np.asarray(heatmap)

    def plot_heatmap(self, image: Image.Image, heatmap: np.ndarray) -> None:
        """
        Plot the Grad-CAM heatmap overlaid on the original image.

        Args:
            - image: The original input image.
            - heatmap: The generated heatmap.
        """
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()

    def explain(self, image_path: str) -> None:
        """
        Generate and visualize the Grad-CAM heatmap for a given image.

        Args:
            - image_path: Path to the input image file.
        """
        # Load and preprocess the image
        image, input_tensor = self._load_and_preprocess_image(image_path)

        # Register hooks
        self._register_hooks(self.target_layer)

        # Forward pass
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        target = output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target.backward()

        # Generate heatmap
        heatmap = self._generate_heatmap(image, self.activations, self.gradients)

        # Plot heatmap
        self.plot_heatmap(image, heatmap)

        # Remove hooks
        self.handle_forward.remove()
        self.handle_backward.remove()
