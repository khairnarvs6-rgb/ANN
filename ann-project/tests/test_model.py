import numpy as np
from src.model import ANN
from src.metrics import accuracy


def test_model_learns_xor_like_data():
    np.random.seed(1)
    X = np.random.randn(400, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
    model = ANN(2, (16, 16), "tanh")
    model.fit(X[:300], y[:300], X[300:], y[300:], epochs=200, batch_size=32,
              lr=0.01, patience=50, verbose=False)
    assert accuracy(y[300:], model.predict(X[300:])) > 0.85


def test_save_and_load(tmp_path):
    m = ANN(4, (8,), "relu")
    p = tmp_path / "w.npz"
    m.save(p)
    m2 = ANN(4, (8,), "relu").load(p)
    X = np.random.randn(3, 4)
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
