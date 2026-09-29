import numpy as np
from .layers import Dense
from .optimizers import get_optimizer


class ANN:
    """Feed-forward neural network for binary classification (sigmoid + BCE)."""

    def __init__(self, n_features, hidden_layers=(32, 16), hidden_activation="relu",
                 dropout=0.0, l2=0.0):
        sizes = [n_features, *hidden_layers]
        self.layers = [Dense(sizes[i], sizes[i + 1], hidden_activation, dropout, l2)
                       for i in range(len(hidden_layers))]
        self.layers.append(Dense(sizes[-1], 1, "sigmoid", dropout=0.0, l2=l2))
        self.l2 = l2

    # ---- core ----
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X

    def backward(self, y_hat, y):
        eps = 1e-12
        da = -(y / (y_hat + eps) - (1 - y) / (1 - y_hat + eps))   # dL/dy_hat
        for layer in reversed(self.layers):
            da = layer.backward(da)

    @staticmethod
    def _bce(y, y_hat):
        eps = 1e-12
        return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))

    def loss(self, y, y_hat):
        reg = 0.5 * self.l2 * sum(np.sum(l.W ** 2) for l in self.layers)
        return self._bce(y, y_hat) + reg

    # ---- API ----
    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y, X_val, y_val, epochs=200, batch_size=32, optimizer="adam",
            lr=0.001, patience=20, verbose=True):
        opt = get_optimizer(optimizer, lr)
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best, best_w, wait = np.inf, None, 0
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                y_hat = self.forward(X[b], training=True)
                self.backward(y_hat, y[b])
                opt.step(self.layers)

            tr_hat, va_hat = self.predict_proba(X), self.predict_proba(X_val)
            tl, vl = self.loss(y, tr_hat), self.loss(y_val, va_hat)
            ta = np.mean((tr_hat >= .5) == y)
            va = np.mean((va_hat >= .5) == y_val)
            for k, v in zip(hist, (tl, vl, ta, va)):
                hist[k].append(float(v))

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:4d} | loss {tl:.4f} | val_loss {vl:.4f} "
                      f"| acc {ta:.4f} | val_acc {va:.4f}")

            if vl < best - 1e-5:
                best, wait = vl, 0
                best_w = [[p.copy() for p in l.params] for l in self.layers]
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_w is not None:                       # restore best weights
            for l, (W, b) in zip(self.layers, best_w):
                l.W, l.b = W, b
        return hist

    # ---- persistence ----
    def save(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrs = {}
        for i, l in enumerate(self.layers):
            arrs[f"W{i}"], arrs[f"b{i}"] = l.W, l.b
        np.savez(path, **arrs)

    def load(self, path):
        d = np.load(path)
        for i, l in enumerate(self.layers):
            l.W, l.b = d[f"W{i}"], d[f"b{i}"]
        return self
