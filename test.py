
from __future__ import print_function

import netCDF4 as nc
import numpy as np
import scipy.ndimage as ndimage
from scipy import signal
import os
import subprocess as sp

from gaussian_filter import gaussian_kernel, convolve, tile_and_reflect

"""
The plan:

- write separate routines in Python for creating the kernel and doing the
  convolution. Compare these to the output from the libraries.
- write equivalent routines in Fortran. Run and compare to above.
- results from Python libraries, Python implementation and Fortran
  implementation are all compared.
"""

def call_make():
    return sp.call(['make'])


def call_f2py():

    ret = 0
    cmd = ['f2py', 'test_interface.F90', '-m', 'test_interface', '-h',
           'test_interface.pyf', '--overwrite-signature']
    ret += sp.call(cmd)

    cmd = ['f2py', '--f90flags=-fdefault-real-8', '-c', 'test_interface.pyf',
           'gaussian_filter.F90', 'test_interface.F90']
    ret += sp.call(cmd)

    return ret


def load_fortran_test_interface():

    try:
        import test_interface as ti
    except ImportError:
        ret = call_f2py()
        assert(ret == 0)
        import test_interface as ti

    return ti.test_interface


def run_fortran_gaussian_filter(sigma, truncate, kx, ky, input, mask=None):

    ti = load_fortran_test_interface()

    if mask is None:
        has_mask = False
        mask = np.ones_like(input)
    else:
        has_mask = True

    k, o = ti.run_gaussian_filter(sigma=sigma, truncate=truncate,
                                  kx=kx, ky=ky,
                                  nx=input.shape[0], ny=input.shape[1],
                                  input=input, mask=mask, has_mask=has_mask)

    return k, o


def run_fortran_tile_and_reflect(input):

    ti = load_fortran_test_interface()
    output = ti.run_tile_and_reflect(input=input)
    return output


class TestFortranFilter():
    """
    Test Fortran implementation.

    Comparisons are made with the Python implementation.
    """

    def __init__(self):
        self.my_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(self.my_dir, 'test_data')

    def test_build(self):
        """
        Test building the Fortran module.
        """

        ret = call_make()
        assert(ret == 0)


    def test_f2py(self):
        """
        Test building the f2py interface.
        """

        ret = call_f2py()
        assert(ret == 0)


    def test_kernel(self):
        """
        Test that the kernel is correct.

        Compare to the Python implementation.
        """

        k_p = gaussian_kernel(1.0, 4.0)
        kx, ky = k_p.shape
        k_f, _ = run_fortran_gaussian_filter(1.0, 4.0, kx, ky,
                                             np.ones((kx, ky)))

        assert((abs(k_p - k_f) < 1e-16).all())


    def test_tile_and_reflect(self):
        """
        Test that Python and Fortran code are doing the tiling in the same way.
        """

        with nc.Dataset(os.path.join(self.data_dir, 'mask.nc')) as f:
            input = f.variables['mask'][:]

        output_f = run_fortran_tile_and_reflect(input)
        output_p = tile_and_reflect(input)

        assert(np.array_equal(output_f, output_p))


    def test_convolve(self):
        """
        Test that convolution routine is correct.

        Compare to Python implementation.
        """

        k_p = gaussian_kernel(1.0, 4.0)
        kx, ky = k_p.shape
        input = np.random.random(size=(100, 100))
        _, output_f = run_fortran_gaussian_filter(1.0, 4.0, kx, ky, input)
        output_p = convolve(input, k_p)

        assert((abs(output_p - output_f) < 1e-15).all())


    def test_filter_with_mask(self):
        """
        Some basic tests with masking.
        """

        input = np.random.random(size=(100, 100))

        # Lots of mask. Recall 0 is masked.
        mask = np.ones_like(input)
        mask[0::2, :] = 0
        _, output = run_fortran_gaussian_filter(1.0, 4.0, 9, 9, input, mask)
        assert(abs(1 - np.sum(output) / np.sum(input)) < 1e-3)

        # No mask - all blur.
        mask = np.ones_like(input)
        _, output = run_fortran_gaussian_filter(1.0, 4.0, 9, 9, input, mask)
        assert(abs(1 - np.sum(output) / np.sum(input)) < 1e-12)

        # All mask - does nothing.
        mask = np.zeros_like(input)
        _, output = run_fortran_gaussian_filter(1.0, 4.0, 9, 9, input, mask)
        assert(np.array_equal(input, output))


    def test_compare_with_mask(self):
        """
        Compare output between Python and Fortran implementations with masking.
        """

        with nc.Dataset(os.path.join(self.data_dir, 'taux.nc')) as f:
            taux_in = np.array(f.variables['taux'][0, :], dtype='float64')

        mask_py = np.zeros_like(taux_in, dtype='bool')
        mask_py[np.where(taux_in == 0)] = True

        mask_f = np.ones_like(taux_in)
        mask_f[np.where(taux_in == 0)] = 0.0

        # Run the scipy version.
        taux_sc = ndimage.gaussian_filter(taux_in, sigma=4.0, truncate=1.0)
        # To do a realistic comparison we need to mask out land points.
        taux_sc = taux_sc * np.logical_not(mask_py)

        # A lower truncation leads to a smaller kernel and hence less guessing
        # in the case of a masked input. This gives a better result for masked
        # inputs.
        k = gaussian_kernel(4.0, truncate=1.0)
        # Run the Python version.
        taux_py = convolve(taux_in, k, mask_py)

        # Run the Fortran version.
        _, taux_f = run_fortran_gaussian_filter(4.0, 1.0, 9, 9, taux_in, mask_f)

        assert(np.max(abs(taux_py - taux_f)) < 1e-14)
        assert(abs(1 - np.sum(taux_in) / np.sum(taux_f)) < 1e-4)


    def test_compare_without_mask(self):
        """
        Compare output between Python and Fortran implementations no masking.
        """

        with nc.Dataset(os.path.join(self.data_dir, 'taux.nc')) as f:
            taux_in = np.array(f.variables['taux'][0, :], dtype='float64')

        # Scipy
        taux_sc = ndimage.gaussian_filter(taux_in, sigma=4.0, truncate=1.0)

        # Run the Python version.
        k = gaussian_kernel(4.0, truncate=1.0)
        taux_py = convolve(taux_in, k)

        _, taux_f = run_fortran_gaussian_filter(4.0, 1.0, 9, 9, taux_in)

        # Check that all implementations are (very) close.
        assert(np.max(abs(taux_sc - taux_py)) < 1e-14)
        assert(np.max(abs(taux_sc - taux_py)) < 1e-14)


class TestPythonFilter():
    """
    Test Python implementation.

    Comparisons with the Scipy packages are made.
    """

    def __init__(self):
        self.my_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(self.my_dir, 'test_data')

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
        with nc.Dataset(os.path.join(self.data_dir, 'taux.nc')) as f:
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

        with nc.Dataset(os.path.join(self.data_dir, 'taux.nc')) as f:
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

        # No mask, blur everything.
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

        with nc.Dataset(os.path.join(self.data_dir, 'taux.nc')) as f:
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
