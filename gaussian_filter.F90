
module gaussian_filter

use, intrinsic :: iso_fortran_env, only: error_unit

implicit none

private

public gaussian_kernel
public convolve
public assert
public tile_and_reflect

contains 

! This is a copy of code found in test/test.py. These two
! implementations must remain equivalent for tests to pass. 
subroutine gaussian_kernel(sigma, kernel, truncate)

    real, intent(in) :: sigma
    real, intent(out), dimension(:,:), allocatable :: kernel
    real, intent(in), optional :: truncate

    real, dimension(:,:), allocatable :: x, y
    integer :: radius, trunc, i, j
    real :: s

    if (present(truncate)) then
        trunc = truncate
    else
        trunc = 4.0
    endif

    radius = int(trunc * sigma + 0.5)
    s = sigma**2

    ! Set up meshgrid.
    allocate(x(-radius:radius, -radius:radius))
    allocate(y(-radius:radius, -radius:radius))
    do j = -radius, radius
        do i = -radius, radius
            x(i, j) = i
            y(i, j) = j
        enddo
    enddo

    ! Make kernel.
    allocate(kernel(-radius:radius, -radius:radius))
    kernel = 2.0*exp(-0.5 * (x**2 + y**2) / s)
    kernel = kernel / sum(kernel)

    deallocate(x)
    deallocate(y)

end subroutine gaussian_kernel

! Set up 3x3 tiles around the input. 
subroutine tile_and_reflect(input, output)

    real, intent(in), dimension(:,:) :: input
    real, intent(out), dimension(:,:), allocatable :: output

    integer :: rows, cols

    rows = ubound(input, 1)
    cols = ubound(input, 2)

    allocate(output(3*rows, 3*cols))

    ! There are 3x3 tiles, we start at the top left and set the tiles up row by
    ! row.
    ! Top left is flipped left-to-right and up-to-down.
    output(:rows, :cols) = input(rows:1:-1, cols:1:-1)
    ! Top centre is flipped up-to-down
    output(:rows, cols+1:2*cols) = input(rows:1:-1, :)
    ! Top right is flipped left-to-right and up-to-down.
    output(:rows, 2*cols+1:3*cols) = input(rows:1:-1, cols:1:-1)
    ! Middle left flipped left-to-right
    output(rows+1:2*rows, :cols) = input(:, cols:1:-1)
    ! Middle centre unchanged
    output(rows+1:2*rows, cols+1:2*cols) = input(:, :)
    ! Middle right flipped left-to-right
    output(rows+1:2*rows, 2*cols+1:3*cols) = input(:, cols:1:-1)
    ! Bottom left flipped left-to-right and up-to-down
    output(2*rows+1:3*rows, :cols) = input(rows:1:-1, cols:1:-1)
    ! Bottom cente flipped up-to-down
    output(2*rows+1:3*rows, cols+1:2*cols) = input(rows:1:-1, :)
    ! Bottom right flipped left-to-right and up-to-down
    output(2*rows+1:3*rows, 2*cols+1:3*cols) = input(rows:1:-1, cols:1:-1)

end subroutine tile_and_reflect

! Convolution. First implement without mask. 
subroutine convolve(input, weights, output, mask)

    real, intent(in), dimension(:,:) :: input, weights
    real, intent(in), dimension(:,:), optional :: mask
    real, intent(inout), dimension(:,:) :: output

    real, dimension(:, :), allocatable :: tiled_input

    integer :: rows, cols, hw_row, hw_col, i, j, tj, ti

    ! First step is to tile the input.
    rows = ubound(input, 1)
    cols = ubound(input, 2)
    ! Stands for half weights row. 
    hw_row = ubound(weights, 1) / 2
    hw_col = ubound(weights, 2) / 2

    ! Only one reflection is done on each side so the weights array cannot be
    ! bigger than width/height of input +1.
    call assert(ubound(weights, 1) < rows + 1, &
                'Input size too small for weights matrix')
    call assert(ubound(weights, 2) < cols + 1, &
                'Input size too small for weights matrix')

    call tile_and_reflect(input, tiled_input)

    do j = 1, cols 
        do i = 1, rows
            ! Use i, j to offset into equivalent part of the tiled_input. 
            tj = j + cols
            ti = i + rows

            ! Find the part of the tiled_input array that overlaps with the
            ! weights array, multiply with weights and sum up. 
            output(i, j) = sum(weights(:,:) * &
                               tiled_input(ti - hw_row:ti + hw_row, &
                                           tj - hw_col:tj + hw_col))
        enddo
    enddo

end subroutine convolve

subroutine assert(statement, msg)

    logical, intent(in) :: statement
    character(len=*), intent(in) :: msg

    if (.not. statement) then 
        write(error_unit, *) msg        
        stop 'Assert triggered, see stderr.'
    endif 

end subroutine assert

end module gaussian_filter
