import io
import numpy as np
import cv2
from PIL import Image

from . import string2Q


def load_qf_map(filename, n_coefs=15):
    qf_list = np.load(filename, allow_pickle=True)
    qf_labels = [qf[:2] for qf in qf_list]
    qf_coeffs = [string2Q(qf[2])[:n_coefs] for qf in qf_list]
    return qf_labels, qf_coeffs


def estimation_by_mse(x, y, weights=None, used_coeffs=slice(16)):
    """
    Estimates QF for given vector of DCT coeffs and mapping of known DCT coeffs to QF estimations

    Args:
        x (numpy.array): vector of DCT coeffs to estimate
        y (numpy.array): QF - DCT mapping
        weights (numpy.array, optional): Weights used for weighted MSE computation. Defaults to None.
        used_coeffs (slice or List[int], optional): Positions of coeffs to be used. Defaults to slice(16).

    Returns:
        numpy.array: (W)MSE values calculated for given DCT coeffs and DCT-QF mapping
    """
    diff = np.asarray((x - y) ** 2)
    return np.average(diff[::, used_coeffs], axis=1, weights=weights[used_coeffs] if weights is not None else weights)


def read_and_preprocess_image(im_file, target_size, scale=255.):
    im = cv2.imread(im_file)
    return preprocess_input(im, target_size, scale)


def preprocess_input(im, target_size, scale=255.):
    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if im.shape != target_size:
        im = cv2.resize(im, target_size)

    return im.astype(np.float32) / scale


def image_by_qf_generator(img_size, qf1, qf2):
    img = np.random.randint(0, 256, img_size)
    img = compress_image(img, qf1)
    img = compress_image(img, qf2)
    return img


def double_compress_image(img, qf1, qf2):
    img = compress_image(img, qf1)
    img = compress_image(img, qf2)
    return img


def compress_image(image, qf):
    jpeg_encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), qf])[1]
    jpeg_encoded_image = Image.open(io.BytesIO(jpeg_encoded))
    return np.array(jpeg_encoded_image)
