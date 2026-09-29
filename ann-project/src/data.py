import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw(csv_path):
    """Load the dataset; downloads (bundled with sklearn) and caches it as CSV on first use."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ds = load_breast_cancer(as_frame=True)
        ds.frame.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


def prepare_data(cfg):
    """Returns dict with scaled train/val/test arrays. Also saves processed splits."""
    df = load_raw(cfg["paths"]["raw_data"])
    X = df.drop(columns="target").values.astype(np.float64)
    y = df["target"].values.astype(np.float64).reshape(-1, 1)

    seed = cfg["seed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=cfg["data"]["val_size"], stratify=y_tr, random_state=seed)

    scaler = StandardScaler().fit(X_tr)          # fit on train only -> no leakage
    X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "splits.npz", X_train=X_tr, y_train=y_tr, X_val=X_va,
             y_val=y_va, X_test=X_te, y_test=y_te,
             mean=scaler.mean_, scale=scaler.scale_)
    return dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
