import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    print(f"Loaded {batch_size}, ({input_height},{input_width}) images with {in_channels} channels ")
    print(f"Loaded")
    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax # 128
    c_out_pmax = nl.tile_size.pmax # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax


    
# - load in the weights into an SBUF array of shape (n_tiles_out_channels, nl.par_dim(c_out_pmax), n_tiles_in_channels, 128, kernel_height, kernel_width)
# - move data around using nl.copy to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
# - transpose that to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w


    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    weights = nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    weights_copy = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    # Reshape input weight matrix and load in sbuf
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))
    weights = nl.load(W)

    # Need to move dimensions around as such
    for height_i in nl.affine_range(filter_height):
        for width_i in nl.affine_range(filter_width):
            for out_i in nl.affine_range(n_tiles_c_out):
                for in_i in nl.affine_range(n_tiles_c_in):
                    for ii in nl.affine_range(c_out_pmax):
                        for jj in nl.affine_range(c_in_pmax):
                            weights_copy[height_i, width_i, out_i, in_i, ii, jj] = nl.copy(weights[out_i, ii, in_i, jj, height_i, width_i])

    # Then transpose
    # weights_copy = nl.transpose(weights_copy)

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # Assign space in sbuf for entire image
        image = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width), 
                        dtype=X.dtype, 
                        buffer=nl.sbuf,
                        )

        # Loop over image and assign each tile
        for k in nl.affine_range(n_tiles_c_in):
            image[k] = nl.load(X[b, (c_in_pmax*k):(c_in_pmax*(k+1)), :,:])

        # Loop over out now
        for k in nl.affine_range(n_tiles_c_out):
            # Assign space to store output in sbuf
            out = nl.ndarray(shape=(nl.par_dim(c_out_pmax), out_height, out_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf)

            # Iterate over output rows
            for j in nl.affine_range(out_height):
                # init row
                out_row = nl.zeros(shape=(c_out_pmax, out_width),
                                    dtype=X.dtype,
                                    buffer=nl.psum)

                for ii in range(filter_height):

                    for jj in range(filter_width):

                        # print(filter_width)
                        for n in nl.affine_range(n_tiles_c_in):

                            out_row[...] = nl.add(out_row,nl.matmul(x = weights_copy[ii, jj, k, n, :, :],
                                                y = image[n, :, j + ii, jj:jj + out_width],
                                                transpose_x =True))

                out[:,j,:] = out_row

        nl.store(X_out[b,:,:,:], value=out)
    return X_out

