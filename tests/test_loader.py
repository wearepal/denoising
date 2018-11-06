import numpy as np
from skimage import io

from common import ROOT_DIR
from utils.loader import HuaweiDataset


def test_load():
    data = HuaweiDataset(root_dir="{}/tests/test_data".format(ROOT_DIR))
    images = next(iter(data))

    clean_image = io.imread("{}/tests/test_data/Category/Clean/Test_Image.png".format(ROOT_DIR))
    noisy_image = io.imread("{}/tests/test_data/Category/Noisy/Test_Image.png".format(ROOT_DIR))

    np.testing.assert_equal(images['clean'], clean_image)
    np.testing.assert_equal(images['noisy'], noisy_image)
