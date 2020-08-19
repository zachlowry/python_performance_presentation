from math import ceil
from typing import Tuple

import numpy as np
import numba as nb
from numba import cuda

# NBINS must be a constant because of CUDA limitations
NBINS = 256


# noinspection PyArgumentList
@cuda.jit(fastmath=True, max_registers=32)
def min_max_kernel(img: np.ndarray,
                   min_max: np.ndarray) -> None:
    """
    Calculates the minimum and maximum pixel values for an image

    Args:
        img (np.ndarray): the image to calculate the minimum and maximum pixel values for
        min_max: (np.ndarray): an output array containing the minimum and maximum pixel values of the image
    """

    pos = cuda.grid(2)
    if pos[0] >= img.shape[0] or pos[1] >= img.shape[1]:
        return

    cuda.atomic.min(min_max, 0, img[pos])
    cuda.atomic.max(min_max, 1, img[pos])


# noinspection PyArgumentList
@cuda.jit(fastmath=True, max_registers=32)
def calculate_image_bins_kernel(img: np.ndarray,
                                min_max: Tuple[int, int],
                                last_bin: int,
                                binned_img: np.ndarray) -> None:
    """
    Calculates the bin index that every pixel should be placed into when calculating the histogram for the image window

    Args:
        img (np.ndarray): the image to calculate the bins for
        min_max (Tuple[int, int]): the minimum and maximum pixel values contained in the image
        last_bin (int): the index of the last bin in the histogram (usually nbins-1)
        binned_img (np.ndarray): aoutput array containing the indices of the bins that the corresponding pixel values
        should be placed into when calculating the histogram for the image window
    """

    # get i, j coordinates for our pixel from the cuda grid routine
    pos = cuda.grid(2)

    # verify that we're within the bounds og the image
    if pos[0] >= img.shape[0] or pos[1] >= img.shape[1]:
        return

    imin, imax = min_max
    binned_img[pos] = max(min((((img[pos] - imin) * NBINS) // (imax - imin)), last_bin), 0)


@cuda.jit(fastmath=True, max_registers=32)
def process_pixel(binned_img: np.ndarray,
                  last_bin: int,
                  window: int,
                  window_mid: int,
                  il_max: int,
                  jl_max: int,
                  window_px: int,
                  clip_limit: int,
                  img_out: np.ndarray) -> None:
    """
    Processes a single image pixel using the Fully-Localized Contrast-Limited Adaptive Histogram Equalization

    Args:
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

    # get i, j coordinates for our pixel from the cuda grid routine
    (i, j) = cuda.grid(2)

    # verify that we're within the bounds og the image
    if i >= binned_img.shape[0] or j >= binned_img.shape[1]:
        return

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
    hist = cuda.local.array(shape=NBINS, dtype=nb.uint16)

    # loop through each pixel in the region
    for ri in range(region.shape[0]):
        for rj in range(region.shape[1]):

            # increment the value for the histogram for this bin
            hist[region[ri, rj]] += 1

    # the pixels clipped to be redistributed across all bins
    excess = 0

    # the calculated look up table value
    lut = 0

    # loop through each histogram bin
    for ii in range(NBINS):

        # if our bin is less than the image pixel intensity, we will increment the "lut"
        if ii <= binned_img[i, j]:

            # if the histogram bin count is greater than clip_limit, add clip limit
            # otherwise add the histogram value to the "lut"
            lut += min(clip_limit, hist[ii])

        # if the histogram bin count is greater than the clip limit, add save the excess
        if hist[ii] > clip_limit:
            excess += (hist[ii] - clip_limit)

    # multiply the lut by the number of histogram bins minus one and add the excess "counts" times the image
    # intensity value then divide by the number of pixels in the window and save the floor
    img_out[i, j] = ((lut * last_bin) + (excess * binned_img[i, j])) // window_px


def flclahe(img: np.ndarray,
            clip_limit: int = 8,
            window: int = 64) -> np.ndarray:
    """
    Processes an image using the Fully-Localized Contrast-Limited Adaptive Histogram Equalization on an image

    Args:
        img (np.ndarray): a 2 dimensional NumPy array containing the image data
        clip_limit (int): the pixel intensity level to use as a threshold for contrast limiting
        window (int): the size in pixels of the NxN window to use when applying the algorithm to the image

    Returns:
        np.ndarray: the resulting processed image data
    """

    # create a stream so we can execute our methods async
    stream = cuda.stream()

    # transfer the image to the device
    d_img = cuda.to_device(img, stream=stream)

    # number of threads per CUDA block
    threads_per_block = (8, 4)

    # calculate the blocks per grid
    blocks_per_grid = (int(ceil(img.shape[0] / threads_per_block[0])),
                       int(ceil(img.shape[1] / threads_per_block[1])))

    # precalculate the index of the last histogram bin
    last_bin = NBINS - 1

    # precalculate the image min and range
    d_min_max = cuda.to_device(np.array([last_bin, 0], dtype=np.uint32), stream=stream)
    min_max_kernel[blocks_per_grid, threads_per_block, stream](d_img,
                                                               d_min_max)

    # create a device array to hold our binned image
    d_binned_img = cuda.device_array_like(img, stream=stream)

    # execute kernel to translate image pixels into histogram bins
    calculate_image_bins_kernel[blocks_per_grid, threads_per_block, stream](d_img,
                                                                            d_min_max,
                                                                            last_bin,
                                                                            d_binned_img)

    # precalculate the midpoint of the window
    window_mid = window // 2

    # precalculate the maximum starting points for windows
    il_max = max(img.shape[0] - window, 0)
    jl_max = max(img.shape[1] - window, 0)

    # precalculate the number of pixels in the window
    window_px = window ** 2

    # scale the provided clip limit by the window size
    img_dtype_iinfo = np.iinfo(img.dtype)
    clip_limit = max(img_dtype_iinfo.min, min(img_dtype_iinfo.max, int(ceil((clip_limit * window_px) / NBINS))))

    # create the output array
    d_img_out = cuda.device_array_like(img, stream=stream)

    # execute kernel to apply FLCLAHE algorithm
    process_pixel[blocks_per_grid, threads_per_block, stream](d_binned_img,
                                                              last_bin,
                                                              window,
                                                              window_mid,
                                                              il_max,
                                                              jl_max,
                                                              window_px,
                                                              clip_limit,
                                                              d_img_out)

    # copy d_img_out to a host array and return
    return d_img_out.copy_to_host()
