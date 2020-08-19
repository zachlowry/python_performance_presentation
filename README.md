This repository walks you through optimizing a Python image processing algorithm from a naive Python-only version that runs for ~500,000 seconds for a 512x512 image into an optimized version that executes in 200ms.

The fully-localized contrast-limited adaptive histogram equalization algorithm was developed by NASA as a way to better visualize thermographical optical data acquired from wind tunnel tests.

More info about FLCLAHE is available from the paper "A Qualitative Investigation of Selected Infrared Flow Visualization
Image Processing Techniques" by Theodore J. Garbeff II and Jennifer K. Baerny, availabile in the 'references' folder. 

Order of evolution:

1.  python - pure python with a variety of "rookie" mistakes
2.  python_refactored - refactored the pure python version to remove obvious errors
3.  numpy_precomputed_bins - precompute the histogram bins for the image
4.  numpy_as_strided - utilize the as_strided feature of NumPy
5.  numpy_bincount - uses np.bincount() as a histogram implementation
6.  numba_skip - skips previously calculated histogram values
7.  numpy_vectorized - uses vectorized NumPy calculations
8.  numpy_threads - uses Python threads
9.  numpy_threads_vectorized - uses Python threads and vectorized NumPy calculations 
10. numba - uses the Numba accelerator
11. numba_skip - uses the Numba accelerator and skips previously calculated histogram values
12. numba_vectorized - uses the Numba accelerator and uses vectorized NumPy calculations
13. numba_parallel - uses the Numba accelerator and Numba's prange() operator
14. numba_threads - uses the Numba accelerator and uses Python threads
15. numba_threads_nogil - uses the Numba accelerator and uses Python threads without the GIL
16. numba_cuda - uses the Numba accelerator and uses a CUDA device for calculation 

Profiling tools used:

 * kernprof
 * memory_profiler
 * gil_load
 * tracemalloc

Todo:

 * Add Cython demo
 * Add CuPy demo
 * Add Pythran example
 * Finish Presentation
