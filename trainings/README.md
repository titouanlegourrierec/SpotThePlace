# `Trainings` module

## Overview

This module provides a template for training an classification model, with custom data processing, and logging functionality. It uses the `Trainer` class to streamline the training, validation, and testing processes and utilizes TensorBoard for performance tracking. The dataset is processed and prepared using custom methods and the `ClassificationDataset` class.

## Folder Strucutre

The project folder contains the following directories and files:

```graphql
trainings/
├── logs/                                       # Directory for training logs
├── runs/                                       # Directory for TensorBoard experiment runs
├── models/                                     # Directory for saving trained models
├── trainings/CNN_FranceGrid_train.py           # Template code for training CNN on France 6*6 grid
data/                                           # Dataset directory
```

## Configuration

Adapt these parameters for the training:
- `BATCH_SIZE`: Batch size used for training and validation.
- `NUM_EPOCHS`: Number of epochs for training.
- `DATASET_PATH`: Path to the dataset.

## Results

After training, TensorBoard can be used to visualize the training progress. You can launch TensorBoard in the directory containing the `trainings/logs/` directory by running:

```bash
tensorboard --logdir=trainings/logs
```
Navigate to http://localhost:6006/ to view the training metrics (loss, accuracy) and graphs.

## Model Saving and Early Stopping

The best model based on validation loss is saved automatically in the `trainings/models/ directory`. If early stopping is enabled, the training will halt if no improvement is seen in the validation loss for a specified number of epochs (`patience` parameter).