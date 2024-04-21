import numpy as np
from mytorch.functional_hw1 import *
from mytorch.functional_hw2 import *

class Downsample1d():

    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Z = A[:, :, ::self.downsampling_factor]

        self.autograd_engine.add_operation(
            [A, np.array([self.downsampling_factor])], 
            Z, 
            [None, None], 
            downsampling1d_backward
        )

        return Z

class Downsample2d():

    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        self.autograd_engine.add_operation(
            [A, np.array([self.downsampling_factor])], 
            Z, 
            [None, None], 
            downsampling2d_backward
        )
        return Z
