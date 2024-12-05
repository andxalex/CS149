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
def fused_conv2d_maxpool2(X,W,bias,pool_size = 1):
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


    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = 128 # 128
    c_out_pmax = 128 # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    X_out = np.zeros(shape = (batch_size, out_channels, out_pool_height, out_pool_width))
    weights = np.zeros(shape = (n_tiles_c_out, c_out_pmax, in_channels_, filter_height, filter_width))
    weights_copy = np.zeros(shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, c_out_pmax, c_in_pmax))

    print(f"Count here X shape is = {X_out.shape}")
    print(f"       Image shape is = {X.shape}")
    print(f"           W shape is = {W.shape}")
    print(f"        bias shape is = {bias.shape}")
    for out_i in range(n_tiles_c_out):
        weights[out_i] = W[out_i * c_out_pmax:(out_i + 1)*c_out_pmax,:,:,:]
    weights= weights.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    # Move
    for height_i in range(filter_height):
        for width_i in range(filter_width):
            for out_i in range(n_tiles_c_out):
                for in_i in range(n_tiles_c_in):
                    weights_copy[height_i, width_i, out_i, in_i, :, :] = weights[out_i, :, in_i, :, height_i, width_i]
                    weights_copy[height_i, width_i, out_i, in_i] = np.transpose(weights_copy[height_i, width_i, out_i, in_i])

    print(f"W shape after is        {weights_copy.shape}")

    for b in range(batch_size):

        # Assign space in sbuf for entire image
        image = np.zeros(shape=(n_tiles_c_in, c_in_pmax, input_height, input_width))

        # Loop over image and assign each tile
        for k in range(n_tiles_c_in):
            image[k] = X[b, (c_in_pmax*k):(c_in_pmax*(k+1)), :,:]

        print(X[b, (c_in_pmax*k):(c_in_pmax*(k+1)), :,:].shape)
        print(image[k].shape)
        # Loop over out now
        for k in range(n_tiles_c_out):
            # Assign space to store output in sbuf
            out = np.zeros(shape=(c_out_pmax, out_height, out_width))
            print(out.shape)
            # Iterate over output rows
            for j in range(out_height):
                # init row
                out_row = np.zeros(shape=(c_out_pmax, out_width))
                # print(f"Out row shape is {out_row.shape}")
                for ii in range(filter_height):

                    for jj in range(filter_width):

                        # print(filter_width)
                        for n in range(n_tiles_c_in):
                            # print(f"Shape of weights {weights_copy[ii, jj, k, n, :, :].shape}")
                            # print(f"Shape of inage {image[n, :, j + ii, jj:jj + out_width].shape}")
                            temp =  weights_copy[ii, jj, k, n, :, :].T @ image[n, :, j + ii, jj:jj + out_width]
                            out_row += temp
                            # print(out_row.shape)
                            # print(f"Accessing element [{n},{150},{j+ii},{jj}:{jj+out_width}]")
                            # print(out_row.shape)
                            # print("??")

                # Write output to sbuf 
                out[:,j,:] = out_row
                


        # Write output to hbm
            print(f"X out shape is {X_out.shape}")
            X_out[b,k*c_out_pmax:(k+1)*c_out_pmax,:,:] = out
        # nl.store(X_out[b,:,:,:], value=out)
        # print(f"Output shape is {X_out.shape}")
        # print(X_out.shape)
    return X_out

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
        shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), in_channels_, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    weights_copy = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    # Reshape input weight matrix and load in sbuf
    print(f"Count here X shape is = {X_out.shape}")
    print(f"       Image shape is = {X.shape}")
    print(f"           W shape is = {W.shape}")
    print(f"        bias shape is = {bias.shape}")
    for out_i in nl.sequential_range(n_tiles_c_out):
        weights[out_i] = nl.load(W[out_i * c_out_pmax:(out_i + 1)*c_out_pmax,:,:,:])
    weights = weights.reshape((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width))
    
    # Need to move dimensions around as such
    for height_i in nl.sequential_range(filter_height):
        for width_i in nl.sequential_range(filter_width):
            for out_i in nl.sequential_range(n_tiles_c_out):
                for in_i in nl.sequential_range(n_tiles_c_in):
                    weights_copy[height_i, width_i, out_i, in_i, :, :] = nl.copy(weights[out_i, :, in_i, :, height_i, width_i])
                    weights_copy[height_i, width_i, out_i, in_i] = nl.transpose(weights_copy[height_i,width_i, out_i, in_i])
                    # for ii in nl.affine_range(c_out_pmax):
                        # for jj in nl.affine_range(c_in_pmax):
                            # weights_copy[height_i, width_i, out_i, in_i, ii, jj] = nl.copy(weights[out_i, ii, in_i, jj, height_i, width_i])

    # print(weights_copy.shape)
    # nl.device_print("here",weights_copy)
    # Then transpose
    # weights_copy = nl.transpose(weights_copy)

    # Process the images in batches
    for b in nl.sequential_range(batch_size):

        # Assign space in sbuf for entire image
        image = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width), 
                        dtype=X.dtype, 
                        buffer=nl.sbuf,
                        )

        # Loop over image and assign each tile
        for k in nl.sequential_range(n_tiles_c_in):
            image[k] = nl.load(X[b, (c_in_pmax*k):(c_in_pmax*(k+1)), :,:])

        print(X[b, (c_in_pmax*k):(c_in_pmax*(k+1)), :,:].shape)
        print(image[k].shape)
        # Loop over out now
        for k in nl.sequential_range(n_tiles_c_out):
            # Assign space to store output in sbuf
            out = nl.zeros(shape=(nl.par_dim(c_out_pmax), out_height, out_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf)

            # Iterate over output rows
            for j in nl.sequential_range(out_height):
                # init row
                out_row = nl.zeros(shape=(c_out_pmax, out_width),
                                    dtype=X.dtype,
                                    buffer=nl.psum)

                for ii in nl.sequential_range(filter_height):

                    for jj in nl.sequential_range(filter_width):

                        # print(filter_width)
                        for n in nl.sequential_range(n_tiles_c_in):
                            temp = nl.matmul(x = weights_copy[ii, jj, k, n, :, :],
                                                y = image[n, :, j + ii, jj:jj + out_width],
                                                transpose_x =True)
                            out_row[...] = nl.add(out_row,temp)
                            print(f"Accessing element [{n},{150},{j+ii},{jj}:{jj+out_width}]")
                            # print(out_row.shape)
                            print("??")

                # Write output to sbuf 
                out[:,j,:] = out_row
                # print(out.shape)

        # Write output to hbm
            nl.store(X_out[b,k*c_out_pmax:(k+1)*c_out_pmax,:,:], value=out)
        # print(f"Output shape is {X_out.shape}")
        # print(X_out.shape)
    return X_out

