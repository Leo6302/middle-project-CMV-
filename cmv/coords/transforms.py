from __future__ import annotations
import numpy as np


def pad_to_3d(pos_2d: np.ndarray) -> np.ndarray:
    """(N,2) → (N,3) with z=0"""
    return np.hstack([pos_2d, np.zeros((len(pos_2d), 1))])
