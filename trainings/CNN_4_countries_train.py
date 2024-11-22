import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spottheplace.utils import data_to_dataframe

from spottheplace.ml import ClassificationDataset
from spottheplace.ml import Trainer


DATASET_PATH = "/Users/titouanlegourrierec/Desktop/dataset_200"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 5

print("Device:", DEVICE)


def configure_model(num_classes, unfreeze_layers=None, feature_extraction=True):
    """
    Configure the ResNet-50 model for either fine-tuning or feature extraction.

    Args:
        num_classes (int): Number of output classes.
        unfreeze_layers (list, optional): Names of layers to unfreeze for fine-tuning. Default is None (freeze all layers).
        feature_extraction (bool): If True, freeze all feature extraction layers.
    
    Returns:
        model: Configured ResNet-50 model.
    """
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if feature_extraction:
        # Freeze all layers if using the model as a feature extractor
        for param in model.parameters():
            param.requires_grad = False

        # Only the head (classification layer) remains trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    elif unfreeze_layers:
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers
        for layer_name in unfreeze_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True

    return model

def main(feature_extraction=True, unfreeze_layers=None):
    """
    Main training function.

    Args:
        feature_extraction (bool): Whether to use the model as a feature extractor.
        unfreeze_layers (list, optional): List of layer names to unfreeze for fine-tuning.
    """
    # Transform the data into a pandas DataFrame
    df = data_to_dataframe(DATASET_PATH)

    # Create the dataset for the model ResNet-50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ClassificationDataset(df, transform=transform, model_type="CNN")

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    trainset, valset, testset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create the dataloaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    # Configure the model
    num_classes = len(dataset.classes)
    model = configure_model(
        num_classes=num_classes,
        unfreeze_layers=unfreeze_layers,
        feature_extraction=feature_extraction
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training and validation
    trainer = Trainer(model, criterion, optimizer, DEVICE)
    trainer.fit(trainloader, valloader, epochs=NUM_EPOCHS)


if __name__ == "__main__":
    # Example 1: Feature extraction only (default behavior)
    # main(feature_extraction=True)

    # Example 2: Fine-tuning with specific layers unfrozen
    # main(feature_extraction=False, unfreeze_layers=["layer4", "fc"])

    # we could try to also unfreeze layer3 to try getting even better performance, but 
    # as we have limited time for the training, we won't retrain it here

    # Choose your desired configuration below
    main(feature_extraction=False, unfreeze_layers=["layer4", "fc"])