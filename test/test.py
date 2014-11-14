
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

class TestFilter():

    def __init__(self):
        self.my_dir = os.path.dirname(os.path.realpath(__file__))

    def test_gaussian_no_mask(self):
        """
        Run the Gaussian filter without a mask and compare to python solution. 
        """

        # Copy input to output. 
        input = os.path.join(self.my_dir, 'taux.nc')
        output = os.path.join(self.my_dir, 'taux_gaussian.nc')

        shutil.copy(input, output)

        with nc.Dataset(input) as f:
            taux_in = f.variables['taux'][0, :]

        taux_out = ndimage.gaussian_filter(taux_in, sigma=3)
