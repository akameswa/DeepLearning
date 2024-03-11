import sys
sys.path.append('mytorch')

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		self.layers = [
			Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
			BatchNorm2d(out_channels), 
			] 										

	def forward(self, A):
		for layer in self.layers:
			A = layer.forward(A)
		return A
	
	def backward(self, grad): 
		for layer in self.layers[::-1]:
			grad = layer.backward(grad)
		return grad

class ResBlock(object):
	def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
		self.convolution_layers = [
			ConvBlock(in_channels, out_channels, filter_size, stride, padding),
			ReLU(),
			ConvBlock(out_channels, out_channels, 1, 1, 0),
			]	
					
		self.final_activation =	ReLU()

		if stride != 1 or in_channels != out_channels or filter_size!=1 or padding!=0:
			self.residual_connection = ConvBlock(in_channels, out_channels, filter_size, stride, padding)
		else:
			self.residual_connection = [Identity()] 


	def forward(self, A):
		Z = A
		'''
		Implement the forward for convolution layer.

		'''
		for l in self.convolution_layers:
			Z = l.forward(Z)
			

		'''
		Add the residual connection to the output of the convolution layers

		'''
		residual = A
		residual = self.residual_connection.forward(residual)

		Z = Z + residual
		

		'''
		Pass the the sum of the residual layer and convolution layer to the final activation function
		'''
		Z = self.final_activation.forward(Z)

		return Z
	

	def backward(self, grad):

		'''
		Implement the backward of the final activation
		'''
		grad = self.final_activation.backward(grad) 


		'''
		Implement the backward of residual layer to get "residual_grad"
		'''
		residual_grad = grad
		residual_grad = self.residual_connection.backward(grad)


		'''
		Implement the backward of the convolution layer to get "convlayers_grad"
		'''
		for l in self.convolution_layers[::-1]:
			grad = l.backward(grad)
			
		convlayers_grad = grad

		'''
		Add convlayers_grad and residual_grad to get the final gradient 
		'''
		grad = convlayers_grad + residual_grad 



		return grad
