import itertools
import os
from math import ceil
from multiprocessing.pool import ThreadPool

import numpy as np
import numba as nb

from .numba import calculate_image_bins
from .numba_threads import _process_pixel

process_pixel = nb.njit(cache=True, fastmath=True, nogil=True)(_process_pixel)


def flclahe(img: np.ndarray,
            clip_limit: int = 8,
            window: int = 64,
            nbins: int = 256) -> np.ndarray:
    """
    Processes an image using the Fully-Localized Contrast-Limited Adaptive Histogram Equalization on an image

    Args:
        img (np.ndarray): a 2 dimensional NumPy array containing the image data
        clip_limit (int): the pixel intensity level to use as a threshold for contrast limiting
        window (int): the size in pixels of the NxN window to use when applying the algorithm to the image
        nbins (int): the number of bins to use when calculation the histogram for the region

    Returns:
        np.ndarray: the resulting processed image data
    """

    # create the output array
    img_out = np.empty_like(img)

    # precalculate the number of pixels in the window
    window_px = window ** 2

    # scale the provided clip limit by the window size
    img_dtype_iinfo = np.iinfo(img.dtype)
    clip_limit = max(img_dtype_iinfo.min, min(img_dtype_iinfo.max, int(ceil((clip_limit * window_px) / nbins))))

    # precalculate the midpoint of the window
    window_mid = window // 2

    # precalculate the maximum starting points for windows
    il_max = max(img.shape[0] - window, 0)
    jl_max = max(img.shape[1] - window, 0)

    # precalculate the image min and range
    imin = np.min(img)
    irng = np.max(img) - imin

    # precalculate the index of the last histogram bin
    last_bin = nbins - 1

    # precalculate all the bin values for the image
    binned_img = calculate_image_bins(img, imin, irng, nbins, last_bin)

    # create a ThreadPool context
    with ThreadPool(processes=os.cpu_count()) as pool:

        # map all of our pixel coordinates to a explicit lambda closure of process_pixel
        pool.map(lambda ij: process_pixel(binned_img,
                                          ij[0],
                                          ij[1],
                                          nbins,
                                          last_bin,
                                          window,
                                          window_px,
                                          window_mid,
                                          il_max,
                                          jl_max,
                                          clip_limit,
                                          img_out),
                 itertools.product(range(img.shape[0]), range(img.shape[1])))

    # return the processed image
    return img_out