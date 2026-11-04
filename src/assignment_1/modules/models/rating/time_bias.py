from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from assignment_1.modules.util import time_bin_edges, time_bins


@dataclass
class TimeBiasFit:
    alpha: float
    users: pd.Index                  # user ID -> position in b_u
    b_u: NDArray[np.float64]
    items: pd.Index                  # item ID -> row in b_i / b_ib
    b_i: NDArray[np.float64]
    b_ib: NDArray[np.float64]        # shape (n_items, n_bins)
    edges: NDArray[np.float64]       # time-bin boundaries from training
    scale: tuple[float, float]


def fit_time_bias(train_df, reg_u, reg_i, reg_ib, n_bins=20, n_iters=10) -> TimeBiasFit:
    r = train_df["rating"].to_numpy(dtype=float)
    # u: user position per row
    u, users = pd.factorize(train_df["user"])
    # i: item position per row
    i, items = pd.factorize(train_df["item"])
    edges = time_bin_edges(train_df["timestamp"], n_bins)
    # actual bins (np.unique may drop some)
    nb = len(edges) - 1
    # one code per (item, bin)
    ib = i * nb + time_bins(train_df["timestamp"], edges)

    n_u = np.bincount(u, minlength=len(users))
    n_i = np.bincount(i, minlength=len(items))
    n_ib = np.bincount(ib, minlength=len(items) * nb)

    alpha = r.mean()
    b_u, b_i, b_ib = np.zeros(len(users)), np.zeros(
        len(items)), np.zeros(len(items) * nb)

    for _ in range(n_iters):
        alpha = np.mean(r - b_u[u] - b_i[i] - b_ib[ib])
        b_u = np.bincount(
            u, weights=r - alpha - b_i[i] - b_ib[ib], minlength=len(users)) / (reg_u + n_u)
        # same pattern: residual without b_i, summed per item, / (reg_i + n_i)
        b_i = np.bincount(
            i, weights=r - alpha - b_u[u] - b_ib[ib], minlength=len(items)) / (reg_i + n_i)
        b_ib = np.bincount(
            ib, weights=r - alpha - b_u[u] - b_i[i], minlength=len(items) * nb) / (reg_ib + n_ib)

    return TimeBiasFit(alpha, users, b_u, items, b_i, b_ib.reshape(len(items), nb), edges,
                       (float(r.min()), float(r.max())))


def predict_time_bias(
    fit: TimeBiasFit,
    users: Iterable[str],
    items: Iterable[str],
    timestamps: pd.Series,
) -> NDArray[np.float64]:
    """Predicted ratings: alpha + b_u + b_i + b_ib[item, bin(t)]; unseen user/item terms are 0."""
    up = fit.users.get_indexer(
        pd.Index(users))       # user positions, -1 if unseen
    # item positions, -1 if unseen
    ip = fit.items.get_indexer(pd.Index(items))
    bins = time_bins(pd.Series(timestamps), fit.edges)

    pred = np.full(len(up), fit.alpha)
    # item term: same pattern with ip and fit.b_i
    pred += np.where(up >= 0, fit.b_u[up], 0.0)            # user term
    pred += np.where(ip >= 0, fit.b_i[ip], 0.0)            # item term
    # item-in-this-time-bin term
    pred += np.where(ip >= 0, fit.b_ib[ip, bins], 0.0)
    return pred.clip(*fit.scale)
