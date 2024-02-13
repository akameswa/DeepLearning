import numpy as np
from mytorch.autograd_engine import Autograd

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
"""

def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""

    return grad_output

def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""
    
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""

    a_grad = grad_output * np.ones(a.shape)
    b_grad = -grad_output * np.ones(b.shape)

    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""

    a_grad = grad_output @ b.T
    b_grad = a.T @ grad_output

    return a_grad, b_grad


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""

    a_grad = grad_output * b
    b_grad = grad_output * a

    return a_grad, b_grad


def div_backward(grad_output, a, b):
    """Backward for division"""

    a_grad = grad_output / b
    b_grad = -grad_output * a / (b ** 2)

    return a_grad, b_grad


def log_backward(grad_output, a):
    """Backward for log"""

    a_grad = grad_output / a

    return a_grad


def exp_backward(grad_output, a):
    """Backward of exponential"""

    a_grad = grad_output * np.exp(a)

    return a_grad


def max_backward(grad_output, a):
    """Backward of max"""

    a_grad = grad_output * np.ones(a.shape)

    return a_grad


def sum_backward(grad_output, a):
    """Backward of sum"""

    a_grad = grad_output * np.ones(a.shape)

    return a_grad

def sigmoid_backward(grad_output, a):
    """Backward of sigmoid"""

    a_grad = grad_output * np.exp(-a) / (1 + np.exp(-a)) ** 2

    return a_grad
def tanh_backward(grad_output, a):
    """Backward of tanh"""

    a_grad = grad_output * (1 - np.tanh(a) ** 2)

    return a_grad
    """
    TODO: implement Softmax CrossEntropy Loss here. You may want to
    modify the function signature to include more inputs.
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """

    return NotImplementedError


