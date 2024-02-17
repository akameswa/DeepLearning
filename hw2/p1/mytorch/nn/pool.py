import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch, channel, width, height = A.shape

        out_width = width - self.kernel + 1
        out_height = height - self.kernel + 1

        Z = np.zeros((batch, channel, out_width, out_height))
        self.maxIdx = np.zeros((batch, channel, out_width, out_height), dtype=object)

        for b in range(batch):
            for c in range(channel):
                for h in range(out_height):
                    for w in range(out_width):
                        a = A[b, c, h:h + self.kernel, w:w + self.kernel]
                        x, y = np.unravel_index(np.argmax(a), a.shape)
                        self.maxIdx[b, c, h, w] = (h + x, w + y)
                        Z[b, c, h, w] = a[x, y]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch, channel, width, height = self.maxIdx.shape
        dLdA = np.zeros(self.A.shape)

        for b in range(batch):
            for c in range(channel):
                for w in range(width):
                    for h in range(height):
                        m_h, m_w = self.maxIdx[b, c, h, w]
                        dLdA[b, c, m_h, m_w] += dLdZ[b, c, h, w]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        raise NotImplementedError

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        raise NotImplementedError


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = None  # TODO
        self.downsample2d = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        raise NotImplementedError

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        raise NotImplementedError


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = None  # TODO
        self.downsample2d = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        raise NotImplementedError

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        raise NotImplementedError
