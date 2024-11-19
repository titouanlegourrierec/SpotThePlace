import os
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class CountryDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images from directories organized by country.
    Each sub-directory represents a country, and images within each directory belong to that country.
    """
    def __init__(self, image_dir, transform=None, num_images_per_country=200):
        self.image_dir = image_dir
        self.transform = transform
        self.num_images_per_country = num_images_per_country
        
        # Get the list of country sub-directories
        self.country_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        
        self.image_files = []
        self.labels = []

        for idx, country in enumerate(self.country_dirs):
            country_path = os.path.join(image_dir, country)
            country_images = [f for f in os.listdir(country_path) if f.endswith('.png')]

            if self.num_images_per_country:
                country_images = random.sample(country_images, min(self.num_images_per_country, len(country_images)))
            
            for img_name in country_images:
                self.image_files.append(os.path.join(country_path, img_name))
                self.labels.append(country)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.class_names = self.label_encoder.classes_

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
