from math import ceil

import numpy as np


def flclahe(
        img: np.ndarray,        # a 2D NumPy array containing the image data
        clip_limit: int = 8,    # the maximum # of pixel counts to use for each pixel histogram bin
        window: int = 64,       # the size in pixels of the square window around the target pixel to use
                                # when calculating the pixel histogram
        nbins: int = 256        # the # of bins to use when calculating the pixel histogram
    ) -> np.ndarray:
    """
    Processes an image using the Fully-Localized Contrast-Limited Adaptive Histogram Equalization on an image

    Args:
        img (np.ndarray): a 2D 8-bit or 16-bit (np.uint8 or np.uint16) NumPy array containing the image data
        clip_limit (int): an optional integer value for the maximum # of pixel counts to use for each pixel histogram
                          bin
        window (int): an optional integer value for the size in pixels of the square window around the target pixel to
                      use when calculating the pixel histogram
        nbins (int): an optional integer value for the # of bins to use when calculating the pixel histogram

    Returns:
        np.ndarray: the resulting processed image data
    """

    # create the output array
    img_out = np.empty_like(img)

    # scale the provided clip limit by the window size
    img_dtype_iinfo = np.iinfo(img.dtype)
    clip_limit = max(img_dtype_iinfo.min, min(img_dtype_iinfo.max, int(ceil((clip_limit * (window ** 2)) / nbins))))

    # loop through each row in the image
    for i in range(img.shape[0]):

        # calculate left index of the window, ensuring that it stays within the image boundaries
        il = i - (window // 2)

        # ensure that the left is never less than 0
        if il < 0:
            il = 0

        # also ensure that the left is not greater than one window size from the edge
        elif il >= img.shape[0] - window:
            il = img.shape[0] - window

        # add the window size to the left index to get the right index
        ir = il + window

        # loop through each column in the image
        for j in range(img.shape[1]):

            # calculate left index of the window, ensuring that it stays within the image boundaries
            jl = j - (window // 2)

            # ensure that the left is never less than 0
            if jl < 0:
                jl = 0

            # also ensure that the left is not greater than one window size from the edge
            elif jl >= img.shape[1] - window:
                jl = img.shape[1] - window

            # add the window size to the left index to get the right index
            jr = jl + window

            # get the window region from the image
            region = img[il:ir, jl:jr]

            # initialize an empty array for the histogram
            hist = np.zeros(shape=(nbins, ), dtype=np.uint16)

            # calculate which bin this pixel should be placed in
            binned_value = ((img[i, j] - img.min()) * nbins) // (img.max() - img.min())

            # ensure that the binned_value is greater than 0
            if binned_value < 0:
                binned_value = 0

            # also ensure that the binned_value is less than nbins
            elif binned_value >= nbins:
                binned_value = nbins - 1

            # loop through each pixel in the region
            for ri in range(region.shape[0]):
                for rj in range(region.shape[1]):

                    # calculate which bin this pixel should be placed in
                    b = ((region[ri, rj] - img.min()) * nbins) // (img.max() - img.min())

                    # ensure that the value of b is greater than 0
                    if b < 0:
                        b = 0

                    # also ensure that the value of b is less than nbins
                    elif b >= nbins:
                        b = nbins - 1

                    # increment the value for the histogram for this bin
                    hist[b] += 1

            # the pixels clipped to be redistributed across all bins
            excess = 0

            # the calculated look up table value
            lut = 0

            # loop through each histogram bin
            for ii in range(nbins):

                # if our bin is less than the image pixel intensity, we will increment the "lut"
                if ii <= binned_value:

                    # if the histogram bin count is greater than clip_limit, add clip limit
                    if hist[ii] > clip_limit:
                        lut += clip_limit

                    # otherwise add the histogram value to the "lut"
                    else:
                        lut += hist[ii]

                # if the histogram bin count is greater than the clip limit, add save the excess
                if hist[ii] > clip_limit:
                    excess += (hist[ii] - clip_limit)

            # start the new value with the value of the "lut"
            new_value = lut

            # multiply the new value by the number of histogram bins minus one
            new_value *= (nbins - 1)

            # add the excess "counts" times the image pixel histogram bin index
            new_value += (excess * binned_value)

            # divide the new value by the number of pixels in the window
            new_value //= (window ** 2)

            # write the new value ot the image
            img_out[i, j] = new_value

    # return the processed image
    return img_out
