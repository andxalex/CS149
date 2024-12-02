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

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Initialize input image tile
    X_tile = nl.ndarray(
        shape=(in_channels, input_height,input_width),
        dtype=X.dtype,
        buffer=nl.sbuf
    )

    W_tile = nl.ndarray(
        shape=(out_channels_, in_channels_, filter_height, filter_width),
        dtype = W.dtype,
        buffer=nl.sbuf
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Weights are constant for all images, reshape here
    # X = X.reshape((batch_size, in_channels, input_height,input_width))

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # Load tile into X and W
        X_tile =nl.load(X[b])
        W_tile =nl.load(W) # this is 128,128,3,3 now, shouldnt it be 128,128?

        # Initialize output with zeros
        temp = nl.zeros((out_channels, input_height*input_width), W.dtype, nl.psum) # PSUM or sbuf?

        for i in range(filter_height):
            for j in range(filter_width):

                # Shift input 
                X_tile = X_tile[:,:,i:,j:]

                # Flatten input
                X_tile = 

                # Perform matmul and accumulate
                # output += matmul(image_shifted, weights[i, j, :, :])

    return X_out

