import pytest
import torch
import torchvision.transforms.functional as TF
from skimage import io
from torch.utils.data import DataLoader
import numpy as np

from common import ROOT_DIR
from utils.inplace_dataset import InplaceHuaweiDataset
import os

from utils.loader import HuaweiDataset
from utils.multi_loaders import create_datasets


def test_load():
    data = HuaweiDataset(root_dir="{}/tests/test_data".format(ROOT_DIR))
    images = next(iter(data))

    clean_image = io.imread("{}/tests/test_data/Category/Clean/Test_image.png".format(ROOT_DIR))
    noisy_image = io.imread("{}/tests/test_data/Category/Noisy/Test_image.png".format(ROOT_DIR))

    np.testing.assert_equal(images['clean'], clean_image)
    np.testing.assert_equal(images['noisy'], noisy_image)


def test_dataset():
    clean_image = io.imread("{}/tests/test_data/Category/Clean/Test_image.png".format(ROOT_DIR))
    noisy_image = io.imread("{}/tests/test_data/Category/Noisy/Test_image.png".format(ROOT_DIR))

    data = InplaceHuaweiDataset(
        root_dir="{}/tests/test_data".format(ROOT_DIR),
        num_crops=1,
        crop_height=clean_image.shape[1],
        crop_width=clean_image.shape[0],
        padding=0)
    images = next(iter(data))

    clean_image = TF.to_tensor(clean_image)
    noisy_image = TF.to_tensor(noisy_image)

    assert torch.all(torch.eq(images['clean'], clean_image))
    assert torch.all(torch.eq(images['noisy'], noisy_image))

    # TODO: PATH Below needs changing
    data = InplaceHuaweiDataset(root_dir=os.path.expanduser("~/Downloads/huawei_ai"), max_idx=5)
    dl = DataLoader(data, batch_size=10)
    for d in dl:
        assert d['clean'].shape == d['noisy'].shape
        assert pytest.approx(d['clean'], d['noisy'])


def test_create_train_test_loaders():

    max_train_samples = 20
    crops_per_image = 3
    ratio = 0.99
    batch_size = 4

    # TODO: PATH Below needs changing
    train_loader, test_loader = create_datasets(os.path.expanduser("~/Downloads/huawei_ai"),
                                                train_ratio=ratio,
                                                max_train_samples=max_train_samples,
                                                crops_per_image=crops_per_image,
                                                batch_size=batch_size)

    assert train_loader.dataset.__len__() == max_train_samples*crops_per_image
    assert test_loader.dataset.__len__() == int(round((1.-ratio)*975))*crops_per_image

    for tr_i in train_loader:
        assert pytest.approx(tr_i['clean'], tr_i['noisy'])

    for te_i in test_loader:
        assert pytest.approx(te_i['clean'], te_i['noisy'])
