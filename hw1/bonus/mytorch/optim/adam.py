# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Adam():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.l = model.layers[::2] # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):
        
        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * layer.dLdW
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (layer.dLdW ** 2)
            
            # TODO: calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.dLdb
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (layer.dLdb ** 2)

            # TODO: Perform weight and bias updates
            m_W_cap = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
            v_W_cap = self.v_W[layer_id] / (1 - self.beta2 ** self.t)

            m_b_cap = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
            v_b_cap = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_cap / (np.sqrt(v_W_cap + self.eps))
            layer.b -= self.lr * m_b_cap / (np.sqrt(v_b_cap + self.eps))
