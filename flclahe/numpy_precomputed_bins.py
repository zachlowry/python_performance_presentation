from math import ceil

import numpy as np


def calculate_image_bins(img: np.ndarray,
                         imin: int,
                         irng: int,
                         nbins: int,
                         last_bin: int) -> np.ndarray:
    """
    Calculates the bin index that every pixel should be placed into when calculating the histogram for the image window

    Args:
        img (np.ndarray): the image to calculate the bins for
        imin (int): the minimum pixel value contained in the image
        irng (int): the difference between the maximum pixel value and the minimum pixel value contained in the image
        nbins (int): the number of bins that will be used in this histogram
        last_bin (int): the index of the last bin in the histogram (usually nbins-1)

    Returns:
        np.ndarray: the indices of the bins that the pixel values should be placed into when calculating the histogram
        for the image window
    """

    binned_img = np.empty_like(img)
    binned_img[:] = np.clip(((img - imin) * nbins) // irng, 0, last_bin)
    return binned_img


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

    # loop through each row in the image
    for i in range(img.shape[0]):

        # calculate left index of the window, ensuring that it stays within the image boundaries
        il = min(max(0, i - window_mid), il_max)

        # add the window size to the left index to get the right index
        ir = il + window

        # loop through each column in the image
        for j in range(img.shape[1]):

            # calculate left index of the window, ensuring that it stays within the image boundaries
            jl = min(max(0, j - window_mid), jl_max)

            # add the window size to the left index to get the right index
            jr = jl + window

            # get the window region from the image
            region = binned_img[il:ir, jl:jr]

            # calculate our histogram using NumPy
            hist = np.bincount(np.ravel(region), minlength=nbins)

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

    # return the processed image
    return img_out
