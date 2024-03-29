import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.r = self.r_act.forward(np.dot(self.Wrx, self.x) + self.brx + np.dot(self.Wrh, self.hidden) + self.brh)
        self.z = self.z_act.forward(np.dot(self.Wzx, self.x) + self.bzx + np.dot(self.Wzh, self.hidden) + self.bzh)
        self.n = self.h_act.forward(np.dot(self.Wnx, self.x) + self.bnx + self.r * (np.dot(self.Wnh, self.hidden) + self.bnh))
        h_t = (1 - self.z) * self.n + self.z * self.hidden

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        x = self.x.reshape(1, self.d)
        hidden = self.hidden.reshape(1, self.h)
        self.r = self.r.reshape(1, self.h)

        self.dLdz = delta * (self.hidden - self.n)
        self.dLdn = delta * (1 - self.z)

        dLdz = self.z_act.backward(self.dLdz).reshape(1, self.h)
        dLdn = self.h_act.backward(self.dLdn).reshape(1, self.h)
        
        self.dWnx = dLdn.T @ x
        self.dbnx = dLdn

        self.dLdr = dLdn * (self.Wnh @ self.hidden + self.bnh).T

        self.dWnh = (dLdn.T * self.r.T) @ hidden    
        self.dbnh = dLdn * self.r

        self.dWzx = dLdz.T @ x
        self.dbzx = dLdz

        self.dWzh = dLdz.T @ hidden
        self.dbzh = dLdz

        dLdr = self.r_act.backward(self.dLdr).reshape(1, self.h)

        self.dWrx = dLdr.T @ x
        self.dbrx = dLdr

        self.dWrh = dLdr.T @ hidden
        self.dbrh = dLdr

        dx = (dLdn @ self.Wnx.reshape(self.h, self.d) + dLdz @ self.Wzx.reshape(self.h, self.d) + dLdr @ self.Wrx.reshape(self.h, self.d) ).reshape(self.d)
        dh_prev_t = (delta * self.z + (dLdn * self.r) @ self.Wnh + dLdz @ self.Wzh + dLdr @ self.Wrh).reshape(self.h)

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t