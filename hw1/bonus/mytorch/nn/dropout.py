# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        self.mask = np.random.binomial(1, 1-self.p, size=x.shape) 

        if train:
            # TODO: Generate mask and apply to x
            return x * self.mask / (1-self.p) 
            
        else:
            # TODO: Return x as is
            return x
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return
        return delta * self.mask