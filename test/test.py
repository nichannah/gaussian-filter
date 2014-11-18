
from __future__ import print_function

import netCDF4 as nc
import numpy as np
import scipy.ndimage as ndimage
from scipy import signal
import shutil
import os

"""
The plan:

- write separate routines in Python for creating the kernel and doing the
  convolution. Compare these to the output from the libraries. 

- write equivalent routines in Fortran. Run and compare to above. 

- results from Python libraries, Python implementation and Fortran
  implementation are all compared. 

"""

def guassian_kernel(sigma, truncate=4.0):
    """
    Return Gaussian that truncates at the given number of standard deviations. 
    """

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    sigma = sigma**2

    k = (1.0 / (2.0*np.pi*sigma))*np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k


def convolve(input, weights):
    """
    2 dimensional convolution.
    
    Borders are handled with reflection. Only one reflection is done on each
    side so the weights array cannot be bigger than width/height of input +1.
    """

    assert(len(input.shape) == 2)
    assert(len(weights.shape) == 2)
    assert(weights.shape[0] < input.shape[0] + 1)
    assert(weights.shape[1] < input.shape[1] + 1)

    rows = input.shape[0]
    cols = input.shape[1]

    input_copy = np.copy(input)
    input = np.tile(input, (3, 3))

    # Now we have a 3x3 tiles - do the reflections. 
    # All those on the sides need to be flipped left-to-right. 
    for i in range(3):
        # Left hand side tiles
        input[i*rows:(i + 1)*rows, 0:cols] = \
            np.fliplr(input[i*rows:(i + 1)*rows, 0:cols])
        # Right hand side tiles
        input[i*rows:(i + 1)*rows, -cols:] = \
            np.fliplr(input[i*rows:(i + 1)*rows, -cols:])

    # All those on the top and bottom need to be flipped up-to-down
    for i in range(3):
        # Top row
        input[0:rows, i*cols:(i + 1)*cols] = \
            np.flipud(input[0:rows, i*cols:(i + 1)*cols])
        # Bottom row
        input[-rows:, i*cols:(i + 1)*cols] = \
            np.flipud(input[-rows:, i*cols:(i + 1)*cols])

    # The central array should be unchanged. 
    assert(np.array_equal(input_copy, input[rows:2*rows, cols:2*cols]))

    # All sides of the middle array should be the same as those bordering them. 
    # Check this starting at the top and going around clockwise. 
    assert(np.array_equal(input_copy[0, :], input[rows-1, cols:2*cols]))
    assert(np.array_equal(input_copy[:, -1], input[rows:2*rows, 2*cols]))
    assert(np.array_equal(input_copy[-1, :], input[2*rows, cols:2*cols]))
    assert(np.array_equal(input_copy[:, 0], input[rows:2*rows, cols-1]))

    # Now do convolution on central array. 


def masked_convolve(input, mask, weights):
    """
    2 dimensional convolution with a mask.
    
    Borders are handled with reflection. 
    """

    assert(False)



class TestFilter():

    def __init__(self):
        self.my_dir = os.path.dirname(os.path.realpath(__file__))

    def test_kernel(self):
        """
        Test that kernel is correct. Compare to one created by ndimage.
        """

        a = np.zeros((9, 9))
        a[4][4] = 1
        k = ndimage.gaussian_filter(a, sigma=1)
        my_k = guassian_kernel(1)
        assert(np.max(abs(my_k - k)) < 1e-16)

        # Also check with a convolution. 
        with nc.Dataset(os.path.join(self.my_dir, 'taux.nc')) as f:
            taux_in = f.variables['taux'][0, :]
        
        taux = ndimage.gaussian_filter(taux_in, sigma=3)
        my_taux = ndimage.convolve(taux_in, guassian_kernel(3))
        assert(np.sum(taux) == np.sum(my_taux))
        assert(np.max(abs(taux - my_taux)) < 1e-6)


    def test_convolve(self):
        """
        Test that convolution routine is correct.
        """

        with nc.Dataset(os.path.join(self.my_dir, 'taux.nc')) as f:
            taux_in = f.variables['taux'][0, :]
        k = guassian_kernel(3)

        with nc.Dataset(os.path.join(self.my_dir, 'mask.nc')) as f:
            mask = f.variables['mask'][:]

        convolve(mask, k)



    def test_gaussian_no_mask(self):
        """
        Run the Gaussian filter without a mask and compare to python solution. 
        """

        # Copy input to output. 
        input = os.path.join(self.my_dir, 'tauy.nc')
        output = os.path.join(self.my_dir, 'tauy_gaussian.nc')

        shutil.copy(input, output)

        with nc.Dataset(input) as f:
            tauy_in = f.variables['tauy'][0, :]

        tauy_out = ndimage.gaussian_filter(tauy_in, sigma=3)

        import pdb
        pdb.set_trace()

        with nc.Dataset(output, 'r+') as f:
            f.variables['tauy'][0, :] = tauy_out
