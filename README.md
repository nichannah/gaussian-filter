
Gaussian filter/blur in Fortran and Python.
===========================================

Edges are treated using reflection. The input can be masked.

This code is being used to smooth out the 'blockiness' which can be seen when doing conservative interpolation of data from coarse to fine grids.

Masking is intended to be conservative and is handled in the following way:

* Masked points are skipped in the convolution, their value will be unchanged.
* Input points which are masked have weight 0 in the kernel. i.e. the kernel is effectively masked.
* The sum of the masked parts of the kernel is evenly distributed over the non-masked part. This ensures that the kernel still has a sum of 1.

There is a fairly extensive set of tests. The different implementations are compared to each other and in some cases also to scipy.ndimage.gaussian_filter

f2py is used by the Python test code to call the Fortran model.

How to Run the tests
--------------------

> nosetests -s

Tests are being run periodically at: https://climate-cms.nci.org.au/jenkins/job/nah599/job/gaussian-filter/

How to Use
-----------

Put gaussian_kernel.F90 into your project.

```
use gaussian_filter, only: gaussian_kernel, convolve

real, dimension(:, :), allocatable :: kernel
real, dimension(x, y) :: input, output

call gaussian_kernel(sigma, kernel, truncate)
call convolve(input, kernel, output, mask)
```

How to Update
-------------

* Make chages to gaussian_kernel.py, add tests to test.py
* Make equivalent changes to gaussian_kernel.F90
* Ensure that the two implementations produce (almost) identical results.
* Consider sending a pull request to add your improvements to this repository.

Example Output
--------------

![Before and after: shortwave from low to high resolution grid](https://raw.github.com/nicholash/gaussian-filter/master/test_data/before_and_after.png)

In this example a shortwave flux from a coarse atmosphere model grid was looking very blocky on the higher resolution ice/ocean grid. This blockiness can cause undesirable artefacts in the ocean outputs. The smoothing has reduced the blockiness. It has introduced a couple of other 'interesting' features:

* Notice the vertical lines, or strips. This pattern corrosponds to the processor tiling, each strip is an individual processor. Since it would be expensive for procs to communicate, each one just smoothes the data that it has available.
* Notice that around land boundaries the smoothing is chunky. This is due to the masking approach which ignores masked areas but increases the weights of nearby non-masked areas (to try to maintain conservation, see explanation of masking above). This probably needs further work.


