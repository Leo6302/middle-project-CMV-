from __future__ import annotations
import numpy as np


def time_split(X: np.ndarray, y: np.ndarray, train_r: float = 0.70, val_r: float = 0.15):
    N = len(X)
    n_train = int(N * train_r)
    n_val = int(N * val_r)
    return (
        X[:n_train], y[:n_train],
        X[n_train:n_train + n_val], y[n_train:n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:],
    )
