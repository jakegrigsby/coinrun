
import numpy as np

from coinrun.main_utils import mpi_print

def slice_spectrum(imgs, high_or_low, r):
    """
    Separate imgs into high and low frequency filters.

    Input is a (NUM_ENVS, 64, 64, 3) image. Channels last!

    We will apply FFT on each channel individually. Then adjust
    the values based on high_or_low and _r. Then inverse FFT back
    to the usable image.
    """
    return imgs
