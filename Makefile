
test.exe: test_interface.F90 gaussian_filter.F90
	gfortran -g -O0 -Wall gaussian_filter.F90 test_interface.F90

gaussian_filter.mod: gaussian_filter.F90
	gfortran -g -O0 -Wall gaussian_filter.F90
