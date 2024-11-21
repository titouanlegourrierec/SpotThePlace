# `ml` module

## Overview

This repository contains two core components:
1. `dataset.py` – Provides classes for loading and preprocessing datasets for classification and regression tasks.
2. `trainer.py` – Simplifies the process of training, validating, and testing a model.

## `dataset.py`

Contains two dataset classes designed for use with ViT (Vision Transformer) or CNN (Convolutional Neural Network) models.

1. `ClassificationDataset`

Used for classification tasks where labels are discrete categories (e.g., countries). Labels are encoded with LabelEncoder.  
* **Input**: DataFrame with image paths and categorical labels.
* **Output**: Preprocessed image and encoded label.

```py
# Example usage of ClassificationDataset
dataset = ClassificationDataset(df, feature_extractor=feature_extractor, model_type='ViT')
or
dataset = ClassificationDataset(df, transform=transform, model_type='CNN')
```


2. `RegressionDataset`

Used for regression tasks where targets are continuous values (e.g., latitude and longitude).
* **Input**: DataFrame with image paths and continuous target values.
* **Output**: Preprocessed image and continuous target.

```py
# Example usage of RegressionDataset
dataset = RegressionDataset(df, feature_extractor=feature_extractor, model_type='ViT')
or
dataset = RegressionDataset(df, transform=transform, model_type='CNN')
```

## `trainer.py`

The `Trainer` class simplifies model training, validation, and testing. It includes several methods to streamline the process:

* `train_one_epoch(dataloader)`: Trains the model for one epoch and returns the average training loss.
* `validate(dataloader)`: Validates the model on a validation dataset and returns the average validation loss.
* `test(dataloader)`: Evaluates the model on test data and returns the loss and predictions.
* `fit(train_loader, val_loader, epochs)`: Trains the model for a specified number of epochs and tracks the training and validation losses.

```py
# Example usage of Trainer class
trainer = Trainer(model, criterion, optimizer, device='cuda')
trainer.fit(train_loader, val_loader, epochs=10)
```

### Other Features
* Early Stopping: The `EarlyStopping` class allows for prematurely stopping training if the validation loss does not improve over a specified number of epochs (defined by the patience parameter).
* TensorBoard Logging: The class uses PyTorch's `SummaryWriter` to log training metrics (loss and accuracy) at each epoch, enabling visualization of the training process in TensorBoard.
