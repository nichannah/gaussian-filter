
! Some interface code to test the gaussian_filter from python. gaussian_filter
! can't be called directly from Python because it has assumed shape arrays. 
module test_interface

use gaussian_filter, only: gaussian_kernel, convolve, assert
    
implicit none

contains

subroutine run_gaussian_filter(sigma, truncate, kx, ky, kernel, &
                               nx, ny, input, output)

    real, intent(in) :: sigma, truncate
    ! Indices and output for kernel
    integer, intent(in) :: kx, ky
    real, intent(out), dimension(kx, ky) :: kernel

    ! Indices and data input/output
    integer, intent(in) :: nx, ny
    real, intent(in), dimension(nx, ny) :: input
    real, intent(out), dimension(nx, ny) :: output

    ! Get the kernel first. 
    real, allocatable, dimension(:,:) :: k

    call gaussian_kernel(sigma, k, truncate)
    call assert(all(shape(k) - shape(kernel) == 0), &
                'Kernel shapes do not match')

    kernel(:, :) = k(:, :)

    call convolve(input, kernel, output)

end subroutine run_gaussian_filter

end module test_interface

program test

    use test_interface, only: run_gaussian_filter

    real, dimension(9, 9) :: kernel
    real, dimension(9, 9) :: input, output

    call random_seed()
    call random_number(input)

    call run_gaussian_filter(1.0, 4.0, 9, 9, kernel, 9, 9, input, output) 
    
end program test
