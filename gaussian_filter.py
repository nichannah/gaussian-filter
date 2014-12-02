
import numpy as np

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
