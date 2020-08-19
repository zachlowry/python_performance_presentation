from math import ceil

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

    # initialize an output array to store the histograms for every unique window
    hists = np.empty(shape=(window_i, window_j, nbins), dtype=np.uint16)
    hists_calculated = np.zeros(shape=(window_i, window_j), dtype=np.bool_)

    # loop through each row in the image
    for i in range(img.shape[0]):

        # calculate our window i index
        wi = min(max(0, i - window_mid), il_max)

        # loop through each column in the image
        for j in range(img.shape[1]):

            # calculate our window j index
            wj = min(max(0, j - window_mid), jl_max)

            # don't recalculate the histogram if we've already calculated it
            if not hists_calculated[wi, wj]:

                # get the window region from the image
                region = windowed_img[wi, wj]

                # calculate our histogram using NumPy
                hists[wi, wj] = np.bincount(np.ravel(region), minlength=nbins)

                # mark this histogram as completed
                hists_calculated[wi, wj] = True

            # get the histogram for this pixel
            hist = hists[wi, wj]

            # the calculated lookup table value
            lut = np.sum(np.clip(hist[:binned_img[i, j] + 1], 0, clip_limit))

            # the pixels clipped to be redistributed across all bins
            excess = np.sum(hist[hist > clip_limit] - clip_limit)

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

    # return the processed image
    return img_out
