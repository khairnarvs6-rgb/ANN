import numpy as np
from src.layers import Dense


def test_dense_shapes():
    layer = Dense(5, 3, "relu")
    out = layer.forward(np.random.randn(4, 5))
    assert out.shape == (4, 3)
    grad = layer.backward(np.ones((4, 3)))
    assert grad.shape == (4, 5)
    assert layer.dW.shape == (5, 3)


def test_gradient_check():
    np.random.seed(0)
    layer = Dense(3, 2, "tanh")
    x = np.random.randn(6, 3)
    loss = lambda: np.sum(layer.forward(x, training=False) ** 2) / (2 * x.shape[0])
    a = layer.forward(x, training=False)
    layer.backward(a)                       # dL/da = a for this loss
    analytic = layer.dW.copy()
    num, eps = np.zeros_like(layer.W), 1e-6
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            layer.W[i, j] += eps; lp = loss()
            layer.W[i, j] -= 2 * eps; lm = loss()
            layer.W[i, j] += eps
            num[i, j] = (lp - lm) / (2 * eps)
    assert np.allclose(analytic, num, atol=1e-6)
