from pathlib import Path
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

class HuaweiDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        """
        Args:
            root_dir (string, optional): Directory with all the image folders, and 'Training_Info.csv'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            # Use the default path, assumes repo was cloned alongside a `data` folder
            self.root_dir = Path(__file__).resolve().parent.parent.parent/"data"
        if not self.root_dir.is_dir():
            raise ValueError("No valid top directory specified")
        self.info_df = pd.read_csv(self.root_dir/"Training_Info.csv")

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if idx not in range(len(self.info_df)):
            raise IndexError
        clean_location, noisy_location = self._get_image_locations(idx)
        clean_image = io.imread(clean_location)
        noisy_image = io.imread(noisy_location)
        ISO = self.info_df.iloc[idx]['ISO_Info']
        sample = {
            'clean': clean_image,
            'noisy': noisy_image,
            'iso': ISO,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    def _get_image_locations(self, idx):
        class_info = self.info_df.iloc[idx]['Class_Info']
        # Class_Info in the csv doesn't match the directory names, so they need fixing:
        if class_info == "building":
            class_dir = "Buildings"
        else:
            class_dir = class_info.capitalize()
        file_name = self.info_df.iloc[idx]['Name_Info'].capitalize()
        clean_location = self.root_dir/class_dir/"Clean"/file_name
        noisy_location = self.root_dir/class_dir/"Noisy"/file_name
        return clean_location, noisy_location
