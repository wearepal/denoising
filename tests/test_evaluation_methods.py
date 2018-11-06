import numpy as np
from skimage import io, measure, color
import torch

from common import ROOT_DIR
from utils.psnr import PSNR
from utils.ssim import SSIM

RTOL = 1e-7


def _load_test_images():
    return [io.imread(f"{ROOT_DIR}/tests/test_data/Category/Clean/Test_image.png")]


def _groundtruth_ssim(img1, img2, multichannel):
    return measure.compare_ssim(img1, img2, multichannel=multichannel, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False, data_range=255)


def _corrupt_image(img):
    return img + 2


def test_psnr():
    """Test torch implementation of the PSNR (peak signal to noise ratio) metric"""
    psnr = PSNR(data_range=255)
    images = _load_test_images()
    for image in images:
        image_c = _corrupt_image(image)
        result = psnr(torch.Tensor(image), torch.Tensor(image_c)).numpy()
        desired = measure.compare_psnr(image, image_c)
        np.testing.assert_allclose(result, desired, rtol=RTOL)


def test_ssim_single_channel():
    """Test torch implementation of the SSIM (structural similarity) metric for grayscale picture"""
    ssim = SSIM(data_range=255, channels=1)
    images = _load_test_images()
    for image in images:
        image = color.rgb2gray(image)
        image_c = _corrupt_image(image)
        # add dimensions for the channel and the batch dimension
        im_tensor = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
        im_tensor_c = torch.Tensor(image_c).unsqueeze(0).unsqueeze(0)

        result = ssim(im_tensor, im_tensor_c).numpy()
        desired = _groundtruth_ssim(image, image_c, multichannel=False)
        np.testing.assert_allclose(result, desired, rtol=RTOL)


def test_ssim_multi_channel():
    """Test torch implementation of the SSIM (structural similarity) metric for color picture"""
    ssim = SSIM(data_range=255, channels=4)
    images = _load_test_images()
    for image in images:
        image_c = _corrupt_image(image)
        # the transpose is necessary to get the structure NCHW instead of NHWC
        im_tensor = torch.Tensor(image).transpose(2, 1).transpose(1, 0).unsqueeze(0)
        im_tensor_c = torch.Tensor(image_c).transpose(2, 1).transpose(1, 0).unsqueeze(0)

        result = ssim(im_tensor, im_tensor_c).numpy()
        desired = _groundtruth_ssim(image, image_c, multichannel=True)
        np.testing.assert_allclose(result, desired, rtol=RTOL)


def test_ssim_single_ch_identical():
    """Test torch implementation of the SSIM (structural similarity) metric for grayscale picture

    Check that comparing identical pictures returns 1
    """
    ssim = SSIM(data_range=255, channels=1)
    images = _load_test_images()
    for image in images:
        image = color.rgb2gray(image)
        # add dimensions for the channel and the batch dimension
        im_tensor = torch.Tensor(image).unsqueeze(0).unsqueeze(0)

        result = ssim(im_tensor, im_tensor).numpy()
        np.testing.assert_allclose(result, 1., rtol=RTOL)


def test_ssim_multi_ch_identical():
    """Test torch implementation of the SSIM (structural similarity) metric for color picture

    Check that comparing identical pictures returns 1
    """
    ssim = SSIM(data_range=255, channels=4)
    images = _load_test_images()
    for image in images:
        # the transpose is necessary to get the structure NCHW instead of NHWC
        im_tensor = torch.Tensor(image).transpose(2, 1).transpose(1, 0).unsqueeze(0)

        result = ssim(im_tensor, im_tensor).numpy()
        np.testing.assert_allclose(result, 1., rtol=RTOL)
