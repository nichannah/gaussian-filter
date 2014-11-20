
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

def gaussian_kernel(sigma, truncate=4.0):
    """
    Return Gaussian that truncates at the given number of standard deviations. 
    """

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    sigma = sigma**2

    k = 2*np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k


def tile_and_reflect(input):
    """
    Make 3x3 tiled array. Central area is 'input', surrounding areas are
    reflected.
    """

    tiled_input = np.tile(input, (3, 3))

    rows = input.shape[0]
    cols = input.shape[1]

    # Now we have a 3x3 tiles - do the reflections. 
    # All those on the sides need to be flipped left-to-right. 
    for i in range(3):
        # Left hand side tiles
        tiled_input[i*rows:(i + 1)*rows, 0:cols] = \
            np.fliplr(tiled_input[i*rows:(i + 1)*rows, 0:cols])
        # Right hand side tiles
        tiled_input[i*rows:(i + 1)*rows, -cols:] = \
            np.fliplr(tiled_input[i*rows:(i + 1)*rows, -cols:])

    # All those on the top and bottom need to be flipped up-to-down
    for i in range(3):
        # Top row
        tiled_input[0:rows, i*cols:(i + 1)*cols] = \
            np.flipud(tiled_input[0:rows, i*cols:(i + 1)*cols])
        # Bottom row
        tiled_input[-rows:, i*cols:(i + 1)*cols] = \
            np.flipud(tiled_input[-rows:, i*cols:(i + 1)*cols])

    # The central array should be unchanged. 
    assert(np.array_equal(input, tiled_input[rows:2*rows, cols:2*cols]))

    # All sides of the middle array should be the same as those bordering them.
    # Check this starting at the top and going around clockwise. This can be
    # visually checked by plotting the 'tiled_input' array.
    assert(np.array_equal(input[0, :], tiled_input[rows-1, cols:2*cols]))
    assert(np.array_equal(input[:, -1], tiled_input[rows:2*rows, 2*cols]))
    assert(np.array_equal(input[-1, :], tiled_input[2*rows, cols:2*cols]))
    assert(np.array_equal(input[:, 0], tiled_input[rows:2*rows, cols-1]))

    return tiled_input


def convolve(input, weights, mask=None, slow=False):
    """
    2 dimensional convolution.

    This is a Python implementation of what will be written in Fortran. 
    
    Borders are handled with reflection.

    Masking is supported in the following way: 
        * Masked points are skipped. 
        * Parts of the input which are masked have weight 0 in the kernel. 
        * Since the kernel as a whole needs to have value 1, the weights of the
          masked parts of the kernel are evenly distributed over the non-masked
          parts. 
    """

    assert(len(input.shape) == 2)
    assert(len(weights.shape) == 2)

    # Only one reflection is done on each side so the weights array cannot be
    # bigger than width/height of input +1.
    assert(weights.shape[0] < input.shape[0] + 1)
    assert(weights.shape[1] < input.shape[1] + 1)

    if mask is not None: 
        # The slow convolve does not support masking. 
        assert(not slow)
        assert(input.shape == mask.shape)
        tiled_mask = tile_and_reflect(mask)

    output = np.copy(input)
    tiled_input = tile_and_reflect(input)

    rows = input.shape[0]
    cols = input.shape[1]
    # Stands for half weights row. 
    hw_row = weights.shape[0] / 2
    hw_col = weights.shape[1] / 2

    # Now do convolution on central array.
    # Iterate over tiled_input. 
    for i, io in zip(range(rows, rows*2), range(rows)):
        for j, jo in zip(range(cols, cols*2), range(cols)):
            # The current central pixel is at (i, j)

            # Skip masked points. 
            if mask is not None and tiled_mask[i, j]:
                continue
            
            average = 0.0
            if slow:
                # Iterate over weights/kernel. 
                for k in range(weights.shape[0]):
                    for l in range(weights.shape[1]):

                        # Get coordinates of tiled_input array that match given
                        # weights 
                        m = i + k - hw_row
                        n = j + l - hw_col

                        average += tiled_input[m, n] * weights[k, l]
            else:
                # Find the part of the tiled_input array that overlaps with the
                # weights array.
                overlapping = tiled_input[i - hw_row:i + hw_row + 1,
                                          j - hw_col:j + hw_col + 1]
                assert(overlapping.shape == weights.shape)
                
                # If any of 'overlapping' is masked then set the corrosponding
                # points in the weights matrix to 0 and redistribute these to
                # non-masked points. 
                if mask is not None:
                    overlapping_mask = tiled_mask[i - hw_row:i + hw_row + 1,
                                                  j - hw_col:j + hw_col + 1]
                    assert(overlapping_mask.shape == weights.shape)

                    # Total value and number of weights clobbered by the mask. 
                    clobber_total = np.sum(weights[overlapping_mask])
                    remaining_num = np.sum(np.logical_not(overlapping_mask))
                    # This is impossible since at least i, j is not masked. 
                    assert(remaining_num > 0)
                    correction = clobber_total / remaining_num

                    # It is OK if nothing is masked - the weights will not be changed.  
                    if correction == 0:
                        assert(not overlapping_mask.any())

                    # Redistribute to non-masked points. 
                    tmp_weights = np.copy(weights)
                    tmp_weights[overlapping_mask] = 0.0
                    tmp_weights[np.where(tmp_weights != 0)] += correction

                    # Should be very close to 1. May not be exact due to rounding.
                    assert(abs(np.sum(tmp_weights) - 1) < 1e-15)

                else:
                    tmp_weights = weights
                    
                merged = tmp_weights[:] * overlapping
                average = np.sum(merged)

            # Set new output value. 
            output[io, jo] = average

    return output


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
        my_k = gaussian_kernel(1)
        assert((abs(my_k - k) < 1e-16).all())

        # Also check with a convolution. 
        with nc.Dataset(os.path.join(self.my_dir, 'taux.nc')) as f:
            taux_in = f.variables['taux'][0, :]
        
        taux = ndimage.gaussian_filter(taux_in, sigma=3)
        my_taux = ndimage.convolve(taux_in, gaussian_kernel(3))
        assert(np.sum(taux) == np.sum(my_taux))
        assert((abs(taux - my_taux) < 1e-6).all())


    def test_convolve_looping(self):
        """
        Test the slow and fast convolution implementations - ensure that they
        are identical. 
        """

        k = gaussian_kernel(3)
        input = np.random.randint(10, size=(50, 50))

        slow_output = convolve(input, k, slow=True)
        fast_output = convolve(input, k)

        # There may be some tiny rounding differences. 
        assert((abs(fast_output - slow_output) < 1e-13).all())


    def test_convolve(self):
        """
        Test that convolution routine is correct.

        Compare to ndimage.convolve. 
        """

        input = np.random.random(size=(100, 100))
        k = gaussian_kernel(1)

        my_output = convolve(input, k)
        output = ndimage.convolve(input, k)

        assert((abs(my_output - output) < 1e-15).all())


    def test_filter_without_mask(self):
        """
        Run the Gaussian filter without a mask and compare to python solution. 
        """

        with nc.Dataset(os.path.join(self.my_dir, 'taux.nc')) as f:
            taux_in = f.variables['taux'][0, :]

        taux = ndimage.gaussian_filter(taux_in, sigma=3)
        my_taux = convolve(taux_in, gaussian_kernel(3))

        assert((abs(taux - my_taux) < 1e-6).all())
        assert(abs(1 - np.sum(taux) / np.sum(my_taux)) < 1e-4)
        assert(abs(1 - np.sum(taux_in) / np.sum(my_taux)) < 1e-4)


    def test_filter_with_mask(self):
        """
        Some basic tests with masking. 
        """

        input = np.random.random(size=(100, 100))
        mask = np.zeros_like(input, dtype='bool')
        mask[0::2, :] = True

        # Lots of mask. 
        result = convolve(input, gaussian_kernel(1), mask)
        assert(abs(1 - np.sum(result) / np.sum(input)) < 1e-3)

        # No mask. 
        mask = np.zeros_like(input, dtype='bool')
        result = convolve(input, gaussian_kernel(1), mask)
        assert(abs(1 - np.sum(result) / np.sum(input)) < 1e-12)

        # All mask - does nothing. 
        mask = np.ones_like(input, dtype='bool')
        result = convolve(input, gaussian_kernel(1), mask)
        assert(np.array_equal(input, result))


    def test_realistic_with_mask(self):
        """
        Test a realistic field and mask. 
        """

        with nc.Dataset(os.path.join(self.my_dir, 'taux.nc')) as f:
            taux_in = f.variables['taux'][0, :]

        mask = np.zeros_like(taux_in, dtype='bool')
        mask[np.where(taux_in == 0)] = True

        taux = ndimage.gaussian_filter(taux_in, sigma=3)
        # To do a realistic comparison we need to mask out land points. 
        taux = taux * np.logical_not(mask)

        # A lower truncation leads to a smaller kernel and hence less guessing
        # in the case of a masked input. This gives a better result for masked
        # inputs. 
        k = gaussian_kernel(4, truncate=1)

        my_taux = convolve(taux_in, k, mask)


    def test_tuning_script(self):
        """
        The tuning script will take an example field in and produce a series of
        plots to help the user decide on a good configuration. 
        """

        pass
        
