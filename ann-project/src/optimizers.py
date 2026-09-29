import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = {}

    def step(self, layers):
        for li, layer in enumerate(layers):
            for pi, (p, g) in enumerate(zip(layer.params, layer.grads)):
                key = (li, pi)
                v = self.v.get(key, np.zeros_like(p))
                v = self.momentum * v - self.lr * g
                self.v[key] = v
                p += v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers):
        self.t += 1
        for li, layer in enumerate(layers):
            for pi, (p, g) in enumerate(zip(layer.params, layer.grads)):
                key = (li, pi)
                m = self.b1 * self.m.get(key, 0) + (1 - self.b1) * g
                v = self.b2 * self.v.get(key, 0) + (1 - self.b2) * g ** 2
                self.m[key], self.v[key] = m, v
                m_hat = m / (1 - self.b1 ** self.t)
                v_hat = v / (1 - self.b2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def get_optimizer(name, lr):
    return {"adam": Adam, "sgd": SGD}[name](lr=lr)
