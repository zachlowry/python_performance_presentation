#!/usr/bin/env python3

import importlib
import time
import hashlib
import os.path
from argparse import ArgumentParser

import numpy as np

from PIL import Image


def run_algorithm(f, img_array, limit=None, hash=None, **kwargs):
    print(f"testing {f.__module__.split('.')[-1]}")

    # record the start time before executing the algorithm
    start_time = time.time()

    # execute the algorithm while applying any given limit
    img_out_array = f(img_array[:limit, :limit], **kwargs)

    # save the elapsed time
    elapsed_time = time.time() - start_time

    # only display the image if we've processed the entire image
    if limit is None:
        img_out = Image.fromarray(img_out_array, 'L')
        img_out.show()

    # initialize our log prefix
    log_prefix = "elapsed time"

    # if we only processed a limited image, then calculate an estimate of what the total calculation time shoul be
    if limit is not None:
        elapsed_time *= (img_array.size / (limit ** 2))
        log_prefix = "estimated elapsed time"

    # calculate the hash of the output image
    h = hashlib.sha1(img_out_array.view(np.uint8)).hexdigest()

    # check if the hash is OK
    hash_ok = h == hash

    # print a log message
    msg = f"{log_prefix} for {f.__module__.split('.')[-1]}: {elapsed_time:0.2f}s"
    if hash is not None:
        msg += f"; hash: {h} ({'ok' if hash_ok else 'bad'})"

    print(msg)

    # return the image array
    return img_out_array


if __name__ == '__main__':
    parser = ArgumentParser(
        description="executes the various implementations of FLCLAHE and prints the processing time"
    )
    parser.add_argument("-i", "--image",
                        choices=("f16", "cosmic_cliffs", "realme"),
                        default="f16",
                        help="the source image file to process")
    parser.add_argument("-c", "--clip-limit",
                        type=int,
                        default=8,
                        help="the FLCLAHE clip limit to use")
    parser.add_argument("-w", "--window",
                        type=int,
                        default=64,
                        help="the FLCLAHE window size to use")
    parser.add_argument("algorithms",
                        nargs="*",
                        default=(
                            "python",
                            "python_refactored",
                            "numpy_precomputed_bins",
                            "numpy_as_strided",
                            "numpy_bincount",
                            "numpy_skip",
                            "numpy_vectorized",
                            "numpy_threads",
                            "numpy_threads_vectorized",
                            "numba",
                            "numba_skip",
                            "numba_vectorized",
                            "numba_threads",
                            "numba_threads_nogil",
                            "numba_parallel",
                            "numba_cuda",
                        ),
                        help="the FLCLAHE algorithm implementations to execute")

    args = parser.parse_args()
    img_path = "images"
    img = Image.open(os.path.join(img_path, f"{args.image}.tif"))
    sha1 = open(os.path.join(img_path, f"{args.image}.sha1")).readline().strip()

    # noinspection PyTypeChecker
    img_array = np.array(img)

    for algorithm in args.algorithms:
        a = importlib.import_module(f"flclahe.{algorithm}")

        if not hasattr(a, 'flclahe'):
            print(f"WARNING: could not locate implementation for algorithm '{algorithm}")
            continue

        run_algorithm(a.flclahe,
                      img_array,
                      hash=sha1,
                      window=args.window,
                      clip_limit=args.clip_limit)

