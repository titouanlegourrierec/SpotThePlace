import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spottheplace.utils import data_to_dataframe
from spottheplace.ml import ClassificationDataset
from spottheplace.ml import Trainer

DATASET_PATH = "./data"  # Path to the dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 20
PATIENCE = 5  # EarlyStopping
RANDOM_STATE = 1234

torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def main():
    # Transform the data into a pandas DataFrame
    df = data_to_dataframe(DATASET_PATH)

    # Create the dataset for the model ViT
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    dataset = ClassificationDataset(df, feature_extractor=feature_extractor, model_type="ViT")
    print({idx: label for idx, label in enumerate(dataset.label_encoder.classes_)})

    # Split the dataset into train, validation and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, persistent_workers=True)

    # Create the model
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(dataset.classes))

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize the Trainer
    trainer = Trainer(model, criterion, optimizer, task_type="classification", device=DEVICE, patience=PATIENCE)

    # Start training and validation
    trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS)

    # Test the model
    test_results = trainer.test(test_loader)
    print(f"Test Loss: {test_results['loss']:.4f}, Test accuracy: {test_results['metric']:.4f}")


if __name__ == "__main__":
    main()
