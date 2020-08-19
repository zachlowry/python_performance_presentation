import itertools
import os
from math import ceil
from multiprocessing.pool import ThreadPool

import numpy as np
import numba as nb

from .numba import calculate_image_bins


def _process_pixel(binned_img: np.ndarray,
                   i: int,
                   j: int,
                   nbins: int,
                   last_bin: int,
                   window: int,
                   window_px: int,
                   window_mid: int,
                   il_max: int,
                   jl_max: int,
                   clip_limit: int,
                   img_out: np.ndarray) -> None:
    """
    Processes a single image pixel using the Fully-Localized Contrast-Limited Adaptive Histogram Equalization

    Args:
        i (int): the i index of the image pixel
        j (int): the j index of the image pixel
        binned_img (np.ndarray): an image-sized array of histogram bin values for each pixel
        last_bin (int): the index of the last bin in the histogram (usually nbins-1)
        window (int): the size in pixels of the NxN window to use when applying the algorithm to the image
        window_mid (int): the precalculated midpoint of the window
        il_max (int): the maximum starting i index that a window can have
        jl_max (int): the maximum starting j index that a window can have
        window_px (int): the number of pixels in a window (window**2)
        clip_limit (int): the pixel intensity level to use as a threshold for contrast limiting
        img_out (np.ndarray): an image-sized output array of histogram bin values for each pixel
    """

    # calculate left index of the window, ensuring that it stays within the image boundaries
    il = min(max(0, i - window_mid), il_max)

    # add the window size to the left index to get the right index
    ir = il + window

    # calculate left index of the window, ensuring that it stays within the image boundaries
    jl = min(max(0, j - window_mid), jl_max)

    # add the window size to the left index to get the right index
    jr = jl + window

    # get the window region from the image
    region = binned_img[il:ir, jl:jr]

    # initialize an empty array for the histogram
    hist = np.zeros(shape=(nbins, ), dtype=np.uint16)

    # loop through each pixel in the region
    for ri in range(region.shape[0]):
        for rj in range(region.shape[1]):

            # calculate which bin this pixel should be placed in
            b = region[ri, rj]

            # increment the value for the histogram for this bin
            hist[b] += 1

    # the pixels clipped to be redistributed across all bins
    excess = 0

    # the calculated look up table value
    lut = 0

    # loop through each histogram bin
    for ii in range(nbins):

        # if our bin is less than the image pixel intensity, we will increment the "lut"
        if ii <= binned_img[i, j]:

            # if the histogram bin count is greater than clip_limit, add clip limit
            # otherwise add the histogram value to the "lut"
            lut += min(clip_limit, hist[ii])

        # if the histogram bin count is greater than the clip limit, add save the excess
        if hist[ii] > clip_limit:
            excess += (hist[ii] - clip_limit)

    # start the new value with the value of the "lut"
    new_value = lut

    # multiply the new value by the number of histogram bins minus one
    new_value *= last_bin

    # add the excess "counts" times the image pixel histogram bin index
    new_value += (excess * binned_img[i, j])

    # divide the new value by the number of pixels in the window
    new_value //= window_px

    # write the new value ot the image
    img_out[i, j] = new_value


process_pixel = nb.njit(cache=True, fastmath=True)(_process_pixel)


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
