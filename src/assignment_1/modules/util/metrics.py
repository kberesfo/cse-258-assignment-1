"""Evaluation metrics for local validation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def calc_mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error (rating tasks 1, 3; likely task 6)."""

    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    return float(np.mean((y_true - y_pred) ** 2))


def calc_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction correct (pairwise tasks 2, 4, 5: did we pick the right item?)."""

    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def calc_bere(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Balanced Error Rate"""
    p = np.asarray(y_pred, dtype=bool)
    l = np.asarray(y_true, dtype=bool)

    TP = np.sum(p & l)
    FP = np.sum(p & ~l)
    TN = np.sum(~p & ~l)
    FN = np.sum(~p & l)

    if (TP + FN) == 0 or (TN + FP) == 0:
        raise ValueError("BER undefined: y_true contains only one class")

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return 1 - 1 / 2 * (TPR + TNR)


def calc_pair_accuracy(
    pairs: pd.DataFrame,
    score_fn: Callable[[ArrayLike, ArrayLike], NDArray[np.float64]],
    cols: tuple[str, str, str, str] = ("user", "item1", "item2", "which"),
) -> float:
    """Pairwise accuracy. cols names the (user, item1, item2, true item) columns."""
    user, item1, item2, which = cols
    users = pairs[user].to_numpy()
    first_score = score_fn(users, pairs[item1].to_numpy())
    second_score = score_fn(users, pairs[item2].to_numpy())

    pred = np.where(
        first_score >= second_score, pairs[item1], pairs[item2]
    )  # ties -> item1
    return float((pred == pairs[which].to_numpy()).mean())
