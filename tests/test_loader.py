import numpy as np
from PIL import Image

from common import ROOT_DIR
from utils import HuaweiDataset


def test_load():
    data = HuaweiDataset(root_dir="{}/tests/test_data".format(ROOT_DIR))
    images = next(iter(data))

    clean_image = Image.open("{}/tests/test_data/Category/Clean/Test_image.png".format(ROOT_DIR))
    noisy_image = Image.open("{}/tests/test_data/Category/Noisy/Test_image.png".format(ROOT_DIR))

    np.testing.assert_equal(images['clean'], clean_image)
    np.testing.assert_equal(images['noisy'], noisy_image)
