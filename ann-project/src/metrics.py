import numpy as np


def accuracy(y, p):
    return float(np.mean(y.ravel() == p.ravel()))


def confusion_matrix(y, p):
    y, p = y.ravel().astype(int), p.ravel().astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def classification_report(y, p):
    cm = confusion_matrix(y, p)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy(y, p), "precision": precision, "recall": recall,
            "f1": f1, "confusion_matrix": cm}
