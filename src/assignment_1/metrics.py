"""Evaluation metrics for local validation."""

import numpy as np


def mse(y_true, y_pred):
    """Mean squared error (rating tasks 1, 3; likely task 6)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy(y_true, y_pred):
    """Fraction correct (pairwise tasks 2, 4, 5: did we pick the right item?)."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
