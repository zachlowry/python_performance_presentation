import itertools
import os
from math import ceil
from multiprocessing.pool import ThreadPool
from typing import Tuple

import numpy as np

from .numpy_precomputed_bins import calculate_image_bins


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

    # calculate the number of windows we will need to compute
    # we want to always compute at least one window, even if il_max / jl_max is 0
    window_i = max(il_max + 1, 1)
    window_j = max(jl_max + 1, 1)

    # calculate the size of the window
    # we want to bound the window size by the image size for cases where the image is smaller than the window
    window_y = min(window, img.shape[0])
    window_x = min(window, img.shape[1])

    # create a "virtual" windowed image
    windowed_img = np.lib.stride_tricks.as_strided(binned_img,
                                                   shape=(window_i,
                                                          window_j,
                                                          window_y,
                                                          window_x),
                                                   strides=(binned_img.strides[0],
                                                            binned_img.strides[1],
                                                            binned_img.strides[0],
                                                            binned_img.strides[1]))

    CHUNK_SIZE = 128

    # initialize an output array to store the histograms for every unique window
    hist = np.empty(shape=(window_i * window_j, nbins),
                    dtype=np.uint16)

    # create this function inside the parent function, this creates an implicit closure around parent scope variables
    def calculate_histogram(coordinates: Tuple[int, int]) -> None:
        """
        Calculates the histogram bin value for this pixel and increments the histogram at that index

        Args:
            coordinates (Tuple[int, int]): the ij index of the image pixel
        """
        # get the i, j positions from the arguments
        i, j = coordinates

        # get the window region from the image
        region = windowed_img[i, j]

        # calculate our histogram using NumPy
        hist[j * window_i + i] = np.bincount(np.ravel(region), minlength=nbins)

    # create a ThreadPool context
    with ThreadPool(processes=os.cpu_count()) as pool:

        # map all of our windows to our function
        pool.map(calculate_histogram, itertools.product(range(window_i),
                                                        range(window_j)))

    # create an index to select the proper histogram for each pixel
    il_array = np.clip(np.arange(img.shape[0]) - window_mid, 0, window_i - 1)
    jl_array = np.clip(np.arange(img.shape[1]) - window_mid, 0, window_j - 1)

    # create this function inside the parent function, this creates an implicit closure around parent scope variables
    def process_chunk(coordinates: Tuple[int, int]) -> None:
        """
        Processes a 2D chunk of pixels with FLCLAHE

        Args:
            coordinates (Tuple[int, int]): the ij index of the starting image pixel
        """

        # get the start_i, start_j positions from the arguments
        start_i, start_j = coordinates

        i_slc = slice(start_i, start_i + CHUNK_SIZE)
        j_slc = slice(start_j, start_j + CHUNK_SIZE)

        # set all the indices less than the image value to true
        indices = np.arange(nbins)

        # create a mask array to select the parts of the histograms we want for each pixel
        mask = np.reshape(
            (np.expand_dims(indices, 0) <= np.expand_dims(np.ravel(binned_img[i_slc, j_slc]), 1)),
            (CHUNK_SIZE, CHUNK_SIZE, -1))

        # multiple i and j together to create an absolute index
        hist_idx = \
            np.ravel((jl_array[j_slc] * window_i) + np.expand_dims(il_array[i_slc], 1))

        # copy the calculated histograms into a new array so they can be indexed by each pixel
        strided_hist = hist[hist_idx].reshape((CHUNK_SIZE, CHUNK_SIZE, nbins))

        # calculate the loop up table value for all pixels
        lut = np.sum(np.clip(strided_hist, 0, clip_limit, where=mask), where=mask, axis=2)

        # the pixels clipped to be redistributed across all bins
        excess = np.sum(strided_hist - clip_limit, where=(strided_hist > clip_limit), axis=2)

        # multiply the lut by the number of histogram bins minus one and add the excess "counts" times the image
        # intensity value then divide by the number of pixels in the window and save the floor
        img_out[i_slc, j_slc] = (((lut * last_bin) + (excess * binned_img[i_slc, j_slc])) // window_px).astype(img.dtype)

    # create a ThreadPool context
    with ThreadPool(processes=os.cpu_count()) as pool:

        # map all of our pixel coordinates to our function
        pool.map(process_chunk, itertools.product(range(0, img.shape[0], CHUNK_SIZE),
                                                  range(0, img.shape[1], CHUNK_SIZE)))

    # return the processed image
    return img_out
