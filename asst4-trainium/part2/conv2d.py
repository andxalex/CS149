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

    # print(f"Loaded {batch_size}, ({input_height},{input_width}) images with {in_channels} channels ")
    # print(f"Loaded")
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
    chunk_size = 16
    n_chunks = (input_height + chunk_size - 1)//chunk_size

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

    bbias = nl.ndarray(
        shape = (c_out_pmax, n_tiles_c_out),
        dtype = bias.dtype,
        buffer = nl.sbuf
    )

    # Reshape input weight matrix and load in sbuf
    # print(f"Count here X shape is = {X_out.shape}")
    # print(f"       Image shape is = {X.shape}")
    # print(f"           W shape is = {W.shape}")
    # print(f"        bias shape is = {bias.shape}")
    # print(f"Number of chunks   is = {n_chunks}")


    # Load weights and unroll
    for out_i in nl.affine_range(n_tiles_c_out):
        bbias[:,out_i] = nl.load(bias[out_i*c_out_pmax:(out_i+1)*c_out_pmax])
        for in_i in nl.affine_range(n_tiles_c_in):
            weights[out_i,:, in_i] = nl.load(W[out_i * c_out_pmax:(out_i + 1)*c_out_pmax,in_i * c_in_pmax:(in_i + 1)*c_in_pmax,:,:])
    # bbias = nl.transpose(bbias)
    # print(bbias.shape)

    # Need to move dimensions around as such
    for height_i in nl.affine_range(filter_height):
        for width_i in nl.affine_range(filter_width):
            for out_i in nl.affine_range(n_tiles_c_out):
                for in_i in nl.affine_range(n_tiles_c_in):
                    weights_copy[height_i, width_i, out_i, in_i, :, :] = nl.copy(weights[out_i, :, in_i, :, height_i, width_i])
                    weights_copy[height_i, width_i, out_i, in_i] = nl.transpose(weights_copy[height_i,width_i, out_i, in_i])

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # Assign space in sbuf for entire image
        input_ch = chunk_size + filter_height - 1
        image = nl.ndarray(shape=(n_chunks,n_tiles_c_in, nl.par_dim(c_in_pmax), input_ch, input_width), 
                        dtype=X.dtype, 
                        buffer=nl.sbuf,
                        )

        # Iterate over chunks first
        for ch in nl.affine_range(n_chunks):
            # and then over tiles
            for k in nl.affine_range(n_tiles_c_in):

                start = ch * input_ch
                end   = (ch + 1) * input_ch

                i_par, i_row, i_col = nl.mgrid[0:c_in_pmax, 0:input_ch, 0: input_width]
                mask = ((ch * chunk_size) + i_row) < input_height
                image[ch, k, ] = nl.load(X[b, (c_in_pmax*k) + i_par, (ch * chunk_size) + i_row, i_col], mask = mask)
                # print(f"Loading chunk {ch}, rows {(ch * input_ch)}:{min(((ch+1) * input_ch), input_height)}")

            # Loop over out now
            for k in nl.affine_range(n_tiles_c_out):
                # Assign space to store output in sbuf
                out = nl.zeros(shape=(nl.par_dim(c_out_pmax), chunk_size, out_width),
                                dtype=X.dtype,
                                buffer=nl.sbuf)

                # Iterate over output rows

                # This loop changes to the output chunk size 
                for j in nl.affine_range(chunk_size):
                    # init row
                    out_row = nl.zeros(shape=(c_out_pmax, out_width),
                                        dtype=np.float32,
                                        buffer=nl.psum)

                    for ii in nl.affine_range(filter_height):

                        for jj in nl.affine_range(filter_width):

                            # print(filter_width)
                            for n in nl.affine_range(n_tiles_c_in):
                                out_row += nl.matmul(x = weights_copy[ii, jj, k, n, :, :],
                                                    y = image[ch, n, :, j + ii, jj:jj + out_width],
                                                    transpose_x =True)
                                 

                    out[:,j,:] = nl.add(out_row, bbias[:,k])


                if (pool_size>1):
                    # Create indices
                    i_0 = nl.arange(c_out_pmax)[:, None, None, None, None]
                    i_1 = nl.arange(chunk_size//pool_size)[None, :, None, None, None]
                    i_2 = nl.arange(pool_size)[None, None, :, None, None]
                    i_3 = nl.arange(out_pool_width)[None, None, None, :, None]
                    i_4 = nl.arange(pool_size)[None, None, None, None, :]

                    out = nl.max(
                        out[i_0, pool_size*i_1 + i_2, pool_size*i_3 + i_4], axis=[2, 4])

                    i_par, i_row, i_col = nl.mgrid[0:c_out_pmax, 0:(chunk_size//pool_size), 0:out_pool_width]
                    mask = ((ch*chunk_size) + i_row*pool_size) < out_height
                    nl.store(X_out[b, (k*c_out_pmax)+i_par, (ch * (chunk_size//pool_size))+i_row, i_col], value=out, mask = mask)
                else:
                    i_par, i_row, i_col = nl.mgrid[0:c_out_pmax, 0:chunk_size, 0:out_width]
                    mask = ((ch*chunk_size) + i_row) < out_height
                    nl.store(X_out[b,(k*c_out_pmax)+i_par,(ch*chunk_size)+i_row,i_col], value=out, mask = mask) 

    return X_out

