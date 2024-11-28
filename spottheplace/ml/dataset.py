from typing import Optional, Tuple, Callable

import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor

from spottheplace.ml.utils import AddMask


class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 feature_extractor: Optional[ViTFeatureExtractor] = None,
                 transform: Optional[Callable] = None,
                 model_type: str = 'ViT') -> None:
        """
        Initializes the Dataset object for classification tasks.

        Args:
            - df (pd.DataFrame): DataFrame containing image paths and labels (countries).
            - feature_extractor (Optional[ViTFeatureExtractor]): Feature extractor for ViT model.
            - transform (Optional[callable]): Transform function for CNN model.
            - model_type (str): Type of model ('ViT' or 'CNN').
        """
        if model_type == 'ViT' and feature_extractor is None:
            raise ValueError("feature_extractor must be provided for ViT model type")
        if model_type == 'CNN' and transform is None:
            raise ValueError("transform must be provided for CNN model type")

        self.df = df
        self.image_paths = self.df['image_path'].values
        self.labels = self.df['label'].values
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.classes = self.label_encoder.classes_
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.model_type = model_type

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple: (image, label)
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = AddMask((0.25, 0.25), (0.15, 0.15))(image)

        if self.model_type == 'ViT':
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            image = inputs["pixel_values"].squeeze().to(dtype=torch.float16)
        elif self.model_type == 'CNN':
            image = self.transform(image).to(dtype=torch.float16)
        else:
            raise ValueError('Model type not supported')

        label = self.labels[idx]

        return image, label


class RegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 feature_extractor: Optional[ViTFeatureExtractor] = None,
                 transform: Optional[Callable] = None,
                 model_type: str = 'ViT') -> None:
        """
        Initializes the Dataset object for regression tasks.

        Args:
            - df (pd.DataFrame): DataFrame containing image paths and targets (longitude and latitude).
            - feature_extractor (Optional[ViTFeatureExtractor]): Feature extractor for ViT model.
            - transform (Optional[callable]): Transform function for CNN model.
            - model_type (str): Type of model ('ViT' or 'CNN').
        """
        if model_type == 'ViT' and feature_extractor is None:
            raise ValueError("feature_extractor must be provided for ViT model type")
        if model_type == 'CNN' and transform is None:
            raise ValueError("transform must be provided for CNN model type")

        self.df = df
        self.image_paths = self.df['image_path'].values

        self.targets = torch.tensor(self.df[['long', 'lat']].values, dtype=torch.float16)

        self.feature_extractor = feature_extractor
        self.transform = transform
        self.model_type = model_type

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple: (image, label)
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = AddMask((0.25, 0.25), (0.15, 0.15))(image)

        if self.model_type == 'ViT':
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            image = inputs["pixel_values"].squeeze().to(dtype=torch.float16)
        elif self.model_type == 'CNN':
            image = self.transform(image).to(dtype=torch.float16)
        else:
            raise ValueError('Model type not supported')

        target = self.targets[idx]

        return image, target
