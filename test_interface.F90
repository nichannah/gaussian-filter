
! Some interface code to test the gaussian_filter from python. gaussian_filter
! can't be called directly from Python because it has assumed shape arrays. Also
! f2py needs kind=8 to use double reals. 
module test_interface

use gaussian_filter, only: gaussian_kernel, convolve, tile_and_reflect, assert

implicit none

private

public run_gaussian_filter, run_tile_and_reflect

contains

subroutine run_gaussian_filter(sigma, truncate, kx, ky, kernel, &
                               nx, ny, input, output, mask)

    real(kind=8), intent(in) :: sigma, truncate
    ! Indices and output for kernel
    integer, intent(in) :: kx, ky
    real(kind=8), intent(out), dimension(kx, ky) :: kernel

    ! Indices and data input/output
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in), dimension(nx, ny) :: input
    real(kind=8), intent(in), dimension(nx, ny), optional :: mask
    real(kind=8), intent(out), dimension(nx, ny) :: output

    ! Get the kernel first.
    real, allocatable, dimension(:,:) :: k

    call gaussian_kernel(sigma, k, truncate)
    call assert(all(shape(k) - shape(kernel) == 0), &
                'Kernel shapes do not match')

    kernel(:, :) = k(:, :)

    if (present(mask)) then
        call convolve(input, kernel, output, mask)
    else
        call convolve(input, kernel, output)
    endif

end subroutine run_gaussian_filter

subroutine run_tile_and_reflect(input, x, y, output)

    integer, intent(in) :: x, y
    real(kind=8), intent(in), dimension(x, y) :: input
    real(kind=8), intent(out), dimension(3*x, 3*y) :: output

    real(kind=8), allocatable, dimension(:, :) :: tmp

    call tile_and_reflect(input, tmp)
    call assert(all(shape(tmp) - shape(output) == 0), &
                'Output shapes do not match')

    output(:,:) = tmp(:,:)

end subroutine run_tile_and_reflect

end module test_interface

!program test

!   use test_interface, only: run_tile_and_reflect

!end program test
