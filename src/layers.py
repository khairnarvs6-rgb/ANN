import numpy as np


# ---------- activations ----------
def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(z.dtype)
def tanh(z):        return np.tanh(z)
def tanh_grad(z):   return 1 - np.tanh(z) ** 2
def sigmoid(z):     return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


class Dense:
    """Fully-connected layer with optional activation and inverted dropout."""

    def __init__(self, n_in, n_out, activation="relu", dropout=0.0, l2=0.0):
        self.activation_name = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.dropout, self.l2 = dropout, l2
        # He init for ReLU, Xavier otherwise
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        self.z = x @ self.W + self.b
        a = self.act(self.z)
        if training and self.dropout > 0:
            self.mask = (np.random.rand(*a.shape) > self.dropout) / (1 - self.dropout)
            a = a * self.mask
        else:
            self.mask = None
        return a

    def backward(self, da):
        if self.mask is not None:
            da = da * self.mask
        dz = da * self.act_grad(self.z)
        m = self.x.shape[0]
        self.dW = self.x.T @ dz / m + self.l2 * self.W
        self.db = dz.sum(axis=0, keepdims=True) / m
        return dz @ self.W.T

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]
