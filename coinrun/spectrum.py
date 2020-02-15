
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from coinrun.main_utils import mpi_print

def create_circular_mask(img, radius):
    h, w = img.shape[-3:-1]
    center = (h/2, w/2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return np.expand_dims(mask, -1)

def slice_spectrum(imgs, high_or_low, r):
    """
    Separate imgs into high and low frequency filters.

    Input is a (NUM_ENVS, 64, 64, 3) image. Channels last!

    We will apply FFT on each channel individually. Then adjust
    the values based on high_or_low and _r. Then inverse FFT back
    to the usable image.
    """
    imgs = imgs.astype(np.float32) / 255.
    if high_or_low == 0:
        # natural image
        return imgs
    else:
        mask = create_circular_mask(imgs, r)
        z = np.fft.fftshift(np.fft.fft2(imgs, axes=(-3, -2)), axes=(-3, -2))
        if high_or_low == -1:
            # keep low frequency features
            z *= mask
        else:
            # keep high frequency features
            z *= np.invert(mask)
        imgs = np.fft.ifft2(np.fft.ifftshift(z, axes=(-3, -2)), axes=(-3, -2)).astype(np.float32)
    return imgs
