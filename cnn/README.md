# Country Classification Project

This project contains code for building, training, and evaluating a deep learning model to classify images into different countries based on the directory structure of the dataset.

## Folder Contents

- dataset.py: Defines a custom PyTorch Dataset class, CountryDataset, for loading and preprocessing images from directories organized by country names.

- main.py: Implements the training and evaluation pipeline. It includes data loading, model initialization, training, validation, and testing steps.

---

## How It Works

### Dataset Preparation

Images are organized in directories, where each directory name corresponds to a country. For example:

```
images/
  ├── France/
  │     ├── img1.png
  │     ├── img2.png
  ├── Japan/
  │     ├── img1.png
  │     ├── img2.png
```

### Main Features

1. **Custom Dataset Loader**:

    - CountryDataset in dataset.py loads images and labels them based on directory names.
    - Supports transformations via the PyTorch transforms module.

2. **Model Training and Evaluation**:

    - Pre-trained ResNet18 from PyTorch is fine-tuned to classify images.
    - The dataset is split into training, validation, and test sets (80%/10%/10%).
    - The model is trained for multiple epochs, and the best weights based on validation accuracy are saved.

3. Image Transformations:

    - Images are resized to 224x224 pixels and normalized for ResNet18.

4. Output Classes:

    - The number of output classes corresponds to the number of unique directories in the dataset.


## Key Functions

- dataset.py

    - CountryDataset:
        - Loads images and encodes labels based on the folder structure.
        - Randomly samples up to num_images_per_country images per country for balanced training.

- main.py

    - initialize_model:
        - Customizes the final layer of ResNet18 for the dataset's classes.
    - train_model:
        - Trains the model with the provided training and validation datasets.
    - evaluate_model:
        - Evaluates the trained model on the test dataset and outputs accuracy.
    - save_model:
        - Saves the trained model to the specified path.


## How to Run

1. Prepare the Dataset:
    - Place images in subdirectories under the images/ folder, organized by country.

2. Install Requirements:
    - Ensure PyTorch, torchvision, and other dependencies are installed.

3. Run the Training Pipeline:

```python main.py```

The script will:

- Load the dataset.
- Train the model.
- Evaluate it on the test set.
- Save the trained model to models/cnn_trained_model.pth.


## Customization

- Number of Images Per Country: Adjust NUM_IMAGES_PER_COUNTRY in main.py.
- Batch Size: Modify batch_size in the DataLoader instances.
- Training Epochs: Change num_epochs in train_model.

## Output

- Logs: Training and validation loss/accuracy per epoch.
- Model: The trained model is saved to the models/ directory.