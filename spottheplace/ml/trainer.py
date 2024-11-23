from datetime import datetime
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu', patience=5, log_dir=None):
        """
        Initializes the Trainer.

        Args:
            - model: The PyTorch model to train.
            - criterion: Loss function to use.
            - optimizer: Optimizer to adjust the model weights.
            - device: Device for training ('cpu' or 'cuda').
            - patience: Number of epochs to wait for improvement before early stopping.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = EarlyStopping(patience=patience)

        if log_dir is None:
            log_dir = f"trainings/runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_one_epoch(self, dataloader, epoch):
        """
        Trains the model for one epoch.

        Args:
            - dataloader: DataLoader for the training data.
            - epoch: Current epoch number.

        Returns:
            - tuple: Average loss and accuracy for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        for inputs, targets in tqdm(dataloader, desc="[Training]"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                    self.writer.add_histogram(f'Weights/{name}', param, epoch)

            self.optimizer.step()

            running_loss += loss.item()
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())

        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_predictions))

        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', accuracy, epoch)

        return avg_loss, accuracy

    def validate(self, dataloader, epoch):
        """
        Validates the model on a validation dataset.

        Args:
            - dataloader: DataLoader for the validation data.

        Returns:
            - tuple: Average loss and accuracy on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="[Validation]")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                all_predictions.append(outputs.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_predictions))

        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        return avg_loss, accuracy

    def test(self, dataloader):
        """
        Tests the model and computes metrics.

        Args:
            - dataloader: DataLoader for the test data.

        Returns:
            - dict: Dictionary containing loss, accuracy, predictions, and targets.
        """
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                all_predictions.append(outputs.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())

        avg_loss = test_loss / len(dataloader)
        accuracy = accuracy_score(torch.cat(all_targets), torch.cat(all_predictions))

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': torch.cat(all_predictions),
            'targets': torch.cat(all_targets)
        }

    def fit(self, train_loader, val_loader, epochs, save_path='trainings/models/model_checkpoint.pth'):
        """
        Trains and validates the model for multiple epochs.

        Args:
            - train_loader: DataLoader for the training data.
            - val_loader: DataLoader for the validation data.
            - epochs: Number of epochs.
            - save_path: Path to save the best model.

        Returns:
            - dict: History of training and validation losses and accuracies.
        """
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            train_loss, train_accuracy = self.train_one_epoch(train_loader, epoch)
            val_loss, val_accuracy = self.validate(val_loader, epoch)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    }, save_path)

            if self.early_stopping.should_stop(val_loss):
                print("Early stopping triggered.")
                break

        print(f"Training completed. Best model saved at {save_path}")
        self.writer.close()
        return history


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def should_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
