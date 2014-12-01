
gaussian_filter.mod: gaussian_filter.F90
	gfortran -g -O0 -Wall -fdefault-real-8 -c gaussian_filter.F90

clean:
	rm -f *.o *.mod *.so *.pyf test_interface-f2pywrappers2.f90  test_interfacemodule.c
