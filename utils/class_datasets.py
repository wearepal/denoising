"""Utilities for loading the dataset one class at a time"""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np

class TransformedClassDataset(Dataset):
    """Class for loading the transformed Huawei dataset"""
    def __init__(self, image_class, root_dir=None, transform=None):
        """
        Args:
            root_dir (string, optional): Directory with all the image folders,
                                         and 'Training_Info.csv'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            # Use the default path, assumes repo was cloned alongside a `data` folder
            self.root_dir = Path(__file__).resolve().parent.parent.parent / "data" / "transformed"
        if not self.root_dir.is_dir():
            raise ValueError("No valid top directory specified")
        full_df = pd.read_csv(self.root_dir / "Training_Info.csv")
        self.info_df = full_df[full_df['Class_Info'] == image_class]
        # Find new 0th original image
        self.offset = int(self.info_df.iloc[0]['Name_Info'].strip(".png").strip("Image_"))
        self.n_originals = len(self.info_df)
        self.patches = len([_ for _ in Path(self.root_dir / "0" / "clean").iterdir()])
        self.len = self.n_originals * self.patches

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if idx > self.len:
            raise IndexError
        clean_location, noisy_location = self._get_image_locations(idx)
        clean_image = Image.open(clean_location)
        noisy_image = Image.open(noisy_location)
        iso = self.info_df.iloc[idx//self.patches]['ISO_Info']
        image_class = self.info_df.iloc[idx//self.patches]['Class_Info']
        sample = {
            'clean': clean_image,
            'noisy': noisy_image,
            'iso': iso,
            'class': image_class
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _get_image_locations(self, idx):
        original_index = idx // self.patches
        patch_index = idx - original_index * self.patches
        clean_location = self.root_dir / str(original_index) / "clean" / f"{patch_index}.png"
        noisy_location = self.root_dir / str(original_index) / "noisy" / f"{patch_index}.png"
        return clean_location, noisy_location
    
    def random_split(self, test_ratio=0.5, data_subset=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_test_images = int(self.n_originals * test_ratio)
        test_original_indices = np.random.choice(np.arange(self.n_originals), n_test_images, replace=False)
        train_original_indices = np.setdiff1d(np.arange(self.n_originals), test_original_indices)

        train_indices = np.concatenate([np.arange(image_no * self.patches, (image_no + 1) * self.patches) for image_no in train_original_indices])
        train_indices = np.random.choice(train_indices, int(len(train_indices)*data_subset), replace=False)

        test_indices = np.concatenate([np.arange(image_no * self.patches, (image_no + 1) * self.patches) for image_no in test_original_indices])
        test_indices = np.random.choice(test_indices, int(len(test_indices)*data_subset), replace=False)

        return Subset(self, train_indices), Subset(self, test_indices)


class TestClassDataset(Dataset):
    """Load test data with ISO information"""
    def __init__(self, folder_path, image_class, transform=None):
        """
        Args:
            folder_path (str): path to folder containing `Testing_Info.csv` and `Testing_Data` dir
            transform (callable, optional): optional transformation function
        """
        self.info_df = pd.read_csv(Path(folder_path).resolve() / "Testing_Info.csv", skiprows=1)
        self.image_folder = Path(folder_path).resolve() / "Testing_Data"
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if idx > len(self.info_df):
            raise IndexError
        noisy_image = Image.open(self.image_folder / self.info_df.iloc[idx]['Name_Info'])
        iso = self.info_df.iloc[idx]['ISO_Info']
        sample = {
            'noisy': noisy_image,
            'iso': iso,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class CsvLoader(Dataset):
    """Load dataset with information from CSV file"""
    def __init__(self, csv_path):
        """
        Args:
            csv_path: (string) path to the CSV file that contains the paths to the images
        """
        full_path = Path(csv_path)
        self.root_path = full_path.parent
        self.info_df = pd.read_csv(full_path)

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if idx > len(self.info_df):
            raise IndexError
        noisy_location = self.root_path / self.info_df.iloc[idx]['noisy_path']
        clean_location = self.root_path / self.info_df.iloc[idx]['clean_path']
        clean_image = Image.open(clean_location)
        noisy_image = Image.open(noisy_location)
        return {
            'clean': transforms.functional.to_tensor(clean_image),
            'noisy': transforms.functional.to_tensor(noisy_image),
            'iso': torch.tensor(self.info_df.iloc[idx]['iso'], dtype=torch.float32),
            'class': self.info_df.iloc[idx]['class']
        }

    def random_split(self, test_ratio=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n_total = self.__len__()
        n_test_images = int(n_total * test_ratio)
        test_idx = np.random.choice(np.arange(n_total), n_test_images, replace=False)
        train_idx = np.setdiff1d(np.arange(n_total), test_idx)
        return Subset(self, train_idx), Subset(self, test_idx)


def transform_class_sample(sample):
    """Transformation for sample dict, should be used for test data as well as train"""
    # Define transforms:
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transformed_sample = {
        'noisy': image_transforms(sample['noisy']),
        'iso': torch.FloatTensor([(sample['iso'] - 1215.32) / 958.13]),   # (x - mean) / std,
    }
    if 'clean' in sample:
        transformed_sample['clean'] = image_transforms(sample['clean'])
    return transformed_sample
