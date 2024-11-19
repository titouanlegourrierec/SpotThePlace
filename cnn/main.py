import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from dataset import CountryDataset  

# Directory containing images for the dataset
IMAGES_DIR = 'images'
NUM_IMAGES_PER_COUNTRY = 200  # Number of images per country for testing (use all images for final run)

MODEL_SAVE_PATH = 'models/cnn_trained_model.pth'

def main():
    """
    Main function to set up the model, data, and training process.
    - Defines image transformation pipeline
    - Loads the dataset
    - Splits the dataset into train, validation, and test sets
    - Initializes the model, loss function, optimizer
    - Trains and evaluates the model
    """
    # Define the image transformations (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet input
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
    ])

    # Instantiate the custom dataset with defined transformations
    dataset = CountryDataset(image_dir=IMAGES_DIR, transform=transform, num_images_per_country=NUM_IMAGES_PER_COUNTRY)

    # Split the dataset into train, validation, and test sets (80% train, 10% validation, 10% test)
    train_size = int(0.8 * len(dataset))
    validation_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - validation_size

    trainset, valset, testset = random_split(dataset, [train_size, validation_size, test_size])

    # Create DataLoader instances for each dataset split
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    valloader = DataLoader(valset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=True)

    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = initialize_model(dataset, device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=3, device=device)

    # Evaluate the model on the test dataset
    evaluate_model(model, testloader, device)

    # Save the trained model
    # save_model(model, MODEL_SAVE_PATH)

def initialize_model(dataset, device):
    """
    Initializes the ResNet model with the appropriate number of output classes.
    
    Parameters:
    - dataset (CountryDataset): The dataset to determine the number of output classes.
    - device (torch.device): The device to load the model onto ('cuda' or 'cpu').

    Returns:
    - model (torch.nn.Module): The initialized ResNet model.
    """
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final fully connected layer to match the number of classes in the dataset
    num_classes = len(dataset.class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)
    return model

def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=25, device='cpu'):
    """
    Train the model using the given training data, loss function, and optimizer.
    
    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - trainloader (DataLoader): The DataLoader for training data.
    - valloader (DataLoader): The DataLoader for validation data.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - num_epochs (int): The number of training epochs.
    - device (str): The device to perform computations on ('cuda' or 'cpu').

    Returns:
    - model (torch.nn.Module): The trained model with the best weights.
    """
    best_model_wts = model.state_dict()  # Store the best weights
    best_acc = 0.0  # Track the best accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Loop through both training and validation phases
        for phase in ['train', 'val']:
            # Set model to train or evaluation mode
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
            else:
                model.eval()  # Set model to evaluation mode
                dataloader = valloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over batches in the current phase
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Zero the gradients for the optimizer

                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients only for training phase
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # Get the predicted class labels
                    loss = criterion(outputs, labels)  # Compute the loss

                    if phase == 'train':
                        loss.backward()  # Backpropagate the gradients
                        optimizer.step()  # Update the model weights

                # Track loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, testloader, device='cpu'):
    """
    Evaluate the trained model on the test dataset and print the accuracy.
    
    Parameters:
    - model (torch.nn.Module): The trained model.
    - testloader (DataLoader): The DataLoader for the test dataset.
    - device (str): The device for computation ('cuda' or 'cpu').
    """
    model.eval()  # Set model to evaluation mode
    running_corrects = 0

    # Disable gradient computation during inference
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the predicted class labels
            running_corrects += torch.sum(preds == labels.data)

    # Compute and print the test accuracy
    accuracy = running_corrects.double() / len(testloader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')


def save_model(model, path):
    """
    Save the trained model to a file.
    
    Parameters:
    - model (torch.nn.Module): The trained model.
    - path (str): The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
