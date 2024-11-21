import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spottheplace.utils import france_grid_to_dataframe

from spottheplace.ml import ClassificationDataset
from spottheplace.ml import Trainer


DATASET_PATH = "/Users/titouanlegourrierec/Desktop/dataset_200"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 5

print("Device:", DEVICE)


def main():
    # Transform the data into a pandas DataFrame
    df_france = france_grid_to_dataframe(os.path.join(DATASET_PATH, "France"), n=6)

    # Create the dataset for the model ResNet-50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ClassificationDataset(df_france, transform=transform, model_type="CNN")

    # Split the dataset into train, validation and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    trainset, valset, testset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create the dataloaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(model, criterion, optimizer, DEVICE)
    trainer.fit(trainloader, valloader, epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
