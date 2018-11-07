import random
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple
import pandas as pd
from utils.loader import HuaweiDataset


def create_datasets(root_d: str,
                    train_ratio: float,
                    max_train_samples: int,
                    crops_per_image: int,
                    batch_size: int,
                    crop_height=64,
                    crop_width=64,
                    padding=None) -> Tuple[DataLoader, DataLoader]:
    pad = padding
    if padding is None:
        pad = crop_height // 2
    root_dir = Path(root_d).resolve()
    info_df = pd.read_csv(root_dir / "Training_Info.csv")
    size = len(info_df)

    indices = list(range(size))
    random.shuffle(indices)

    train_indices = indices[:int(round(size*train_ratio))]
    train_indices = train_indices[:max_train_samples]
    test_indices = indices[int(round(size*train_ratio)):]

    train_dataset = HuaweiDataset(root_dir=root_dir, indices=train_indices,
                                  max_idx=len(train_indices), num_crops=crops_per_image,
                                  crop_height=crop_height, crop_width=crop_width,
                                  padding=pad)
    test_dataset = HuaweiDataset(root_dir=root_dir, indices=test_indices,
                                 max_idx=len(test_indices), num_crops=crops_per_image,
                                 crop_height=crop_height, crop_width=crop_width,
                                 padding=pad)

    return DataLoader(train_dataset, batch_size=batch_size), \
           DataLoader(test_dataset, batch_size=batch_size)
