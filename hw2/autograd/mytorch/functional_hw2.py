import numpy as np
from mytorch.autograd_engine import Autograd


def conv1d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    kernel_size = weight.shape[2]
    output_size = A.shape[2] - kernel_size + 1

    dLdb = np.sum(dLdZ, axis=(0,2))
    dLdW = np.zeros_like(weight)

    for i in range(kernel_size):
        row = A[:,:,i:i+output_size]
        dLdW[:,:,i] = np.tensordot(dLdZ, row, axes=([0, 2], [0, 2]))

    dLdA = np.zeros_like(A)
    padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0), (kernel_size-1, kernel_size-1)))
    flipped_weight = np.flip(weight, axis=2)

    for i in range(A.shape[2]):
        dLdA[:,:,i] = np.tensordot(padded_dLdZ[:,:,i:i+kernel_size], flipped_weight, axes=([1,2],[0,2]))

    return dLdA, dLdW, dLdb


def conv2d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    kernel_size = weight.shape[2]
    output_height = A.shape[2] - kernel_size + 1
    output_width = A.shape[3] - kernel_size + 1
    
    dLdb = np.sum(dLdZ, axis=(0,2,3))
    dLdW = np.zeros_like(weight)

    for i in range(kernel_size):
        for j in range(kernel_size):
            row = A[:,:,i:i+output_height,j:j+output_width]
            dLdW[:,:,i,j] = np.tensordot(dLdZ, row, axes=([0, 2, 3], [0, 2, 3]))

    dLdA = np.zeros_like(A)
    padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0), (kernel_size-1, kernel_size-1), (kernel_size-1, kernel_size-1)))
    flipped_weight = np.flip(weight, axis=(2,3))

    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            dLdA[:,:,i,j] = np.tensordot(padded_dLdZ[:,:,i:i+kernel_size,j:j+kernel_size], flipped_weight, axes=([1,2,3],[0,2,3]))

    return dLdA, dLdW, dLdb


def downsampling1d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], A.shape[2]))
    dLdA[:, :, ::int(downsampling_factor[0])] = dLdZ

    return dLdA, None, None


def downsampling2d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], A.shape[2], A.shape[3]))
    dLdA[:, :, ::int(downsampling_factor[0]), ::int(downsampling_factor[0])] = dLdZ

    return dLdA, None, None
