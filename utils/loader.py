"""Utilities for loading the dataset"""
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms
from PIL import Image


class HuaweiDataset(Dataset):
    """Class for loading the Huawei dataset"""
    def __init__(self,
                 root_dir=None,
                 indices=None,
                 transform=None,
                 num_crops=3,
                 crop_height=64,
                 crop_width=64,
                 padding=None,
                 max_idx=None):
        """
        Args:
            root_dir (string, optional): Directory with all the image folders,
                                         and 'Training_Info.csv'.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_crops (optional): Number of crops to apply to an image
            crop_height (optional): Height of the random crop
            crop_width (optional): Width of the random crop
            padding: number of pixels of uniform padding to be applied to all borders
            max_idx (optional): sets the max number of images, mostly to reduce testing time

        """
        self.transform = transform
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        if padding is None:
            self.pad = self.crop_height//2
        else:
            self.pad = padding
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            # Use the default path, assumes repo was cloned alongside a `data` folder
            self.root_dir = Path(__file__).resolve().parent.parent.parent / "data"
        if not self.root_dir.is_dir():
            raise ValueError("No valid top directory specified")
        self.info_df = pd.read_csv(self.root_dir / "Training_Info.csv")
        if max_idx is None:
            self.max_idx = len(self.info_df)
        else:
            self.max_idx = max_idx
        if indices is not None:
            self.indices = [i for x in range(self.num_crops) for i in indices]
        else:
            self.indices = None

    def __len__(self):
        return self.max_idx*self.num_crops

    def __getitem__(self, index):
        if index not in range(len(self.info_df)):
            raise IndexError

        if self.indices is not None:
            idx = self.indices[index]
        else:
            idx = index

        clean_location, noisy_location = self._get_image_locations(idx)

        clean_image = Image.open(clean_location)
        clean_image = torchvision.transforms.functional.pad(clean_image, self.pad)
        noisy_image = Image.open(noisy_location)
        noisy_image = torchvision.transforms.functional.pad(noisy_image, self.pad)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(clean_image, output_size=(self.crop_width, self.crop_height))

        clean_image = TF.crop(clean_image, i, j, h, w)
        noisy_image = TF.crop(noisy_image, i, j, h, w)

        iso = self.info_df.iloc[idx]['ISO_Info']
        sample = {
            'clean': torchvision.transforms.functional.to_tensor(clean_image),
            'noisy': torchvision.transforms.functional.to_tensor(noisy_image),
            'iso': iso,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _get_image_locations(self, idx):
        img_idx = idx//self.num_crops

        class_info = self.info_df.iloc[img_idx]['Class_Info']
        # Class_Info in the csv doesn't match the directory names, so they need fixing:
        if class_info == "building":
            class_dir = "Buildings"
        else:
            class_dir = class_info.capitalize()
        file_name = self.info_df.iloc[img_idx]['Name_Info'].capitalize()
        clean_location = self.root_dir / class_dir / "Clean" / file_name
        noisy_location = self.root_dir / class_dir / "Noisy" / file_name
        return clean_location, noisy_location
