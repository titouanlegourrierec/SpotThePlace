from datetime import datetime
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, criterion, optimizer, task_type='classification', device='cpu', patience=5, log_dir=None):
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
        self.task_type = task_type
        self.device = device
        self.scaler = torch.amp.GradScaler()  # For mixed-precision training
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

            # Forward and backward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()  # Scale loss for stability
            self.scaler.step(self.optimizer)  # Step optimizer
            self.scaler.update()  # Update the scale for next iteration

            running_loss += loss.item()
            if self.task_type == 'classification':
                all_predictions.append(outputs.argmax(dim=1).cpu())
            else:
                all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

        avg_loss = running_loss / len(dataloader)
        if self.task_type == 'classification':
            metric = accuracy_score(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())
            self.writer.add_scalar('Accuracy/Train', metric, epoch)
        else:
            metric = mean_squared_error(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())
            self.writer.add_scalar('MSE/Train', metric, epoch)

        self.writer.add_scalar('Loss/Train', avg_loss, epoch)

        return avg_loss, metric

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

                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                if self.task_type == 'classification':
                    all_predictions.append(outputs.argmax(dim=1).cpu())
                else:
                    all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / len(dataloader)
        if self.task_type == 'classification':
            metric = accuracy_score(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())
            self.writer.add_scalar('Accuracy/Validation', metric, epoch)
        else:
            metric = mean_squared_error(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())
            self.writer.add_scalar('MSE/Validation', metric, epoch)

        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)

        return avg_loss, metric

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

                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                if self.task_type == 'classification':
                    all_predictions.append(outputs.argmax(dim=1).cpu())
                else:
                    all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = test_loss / len(dataloader)
        if self.task_type == 'classification':
            metric = accuracy_score(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())
        else:
            metric = mean_squared_error(torch.cat(all_targets).detach().numpy(), torch.cat(all_predictions).detach().numpy())

        return {
            'loss': avg_loss,
            'metric': metric,
            'predictions': torch.cat(all_predictions),
            'targets': torch.cat(all_targets)
        }

    def fit(self, train_loader, val_loader, epochs, save_path='trainings/models/model.pth'):
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
        history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            train_loss, train_metric = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metric = self.validate(val_loader, epoch)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_metric'].append(train_metric)
            history['val_metric'].append(val_metric)

            if self.task_type == 'classification':
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_metric:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_metric:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train MSE: {train_metric:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_metric:.4f}")

            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)

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
