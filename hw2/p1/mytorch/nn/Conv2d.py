import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        self.output_height = A.shape[2] - self.kernel_size + 1
        self.output_width = A.shape[3] - self.kernel_size + 1

        Z = np.zeros((A.shape[0], self.out_channels, A.shape[2] - self.kernel_size + 1, A.shape[3] - self.kernel_size + 1))
        
        for i in range(Z.shape[2]):
            for j in range(Z.shape[3]):
                Z[:,:,i,j] = np.tensordot(A[:, :, i:i + self.kernel_size, j:j + self.kernel_size], self.W, axes=([1,2,3],[1,2,3])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        self.dLdb = np.sum(dLdZ, axis=(0,2,3))
        self.dLdW = np.zeros_like(self.W)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                row = self.A[:,:,i:i+self.output_height,j:j+self.output_width]
                self.dLdW[:,:,i,j] = np.tensordot(dLdZ, row, axes=([0, 2, 3], [0, 2, 3]))

        self.dLdA = np.zeros_like(self.A)
        padded_dLdZ = np.pad(dLdZ, ((0,0),(0,0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)))
        flipped_weight = np.flip(self.W, axis=(2,3))

        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                self.dLdA[:,:,i,j] = np.tensordot(padded_dLdZ[:,:,i:i+self.kernel_size,j:j+self.kernel_size], flipped_weight, axes=([1,2,3],[0,2,3]))

        return self.dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
