from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from assignment_1.modules.util import time_bin_edges, time_bins

from .base import RatingModel


class TimeBiasModel(RatingModel):
    """
    Bias model with a per-item, per-time-bin offset, fit by ALS (closed-form updates via
    np.bincount). Time bins are quantiles of the training timestamps. Needs a timestamp column.
    Unseen users / items contribute 0 for their terms.
    """

    def __init__(self, reg_u: float, reg_i: float, reg_ib: float,
                 n_bins: int = 20, n_iters: int = 10) -> None:
        self.reg_u, self.reg_i, self.reg_ib = reg_u, reg_i, reg_ib
        self.n_bins, self.n_iters = n_bins, n_iters
        # fitted state (set by fit)
        self.alpha: float = 0.0
        self.users: pd.Index | None = None       # user ID -> position in b_u
        self.items: pd.Index | None = None       # item ID -> row in b_i / b_ib
        self.b_u = self.b_i = self.b_ib = None   # b_ib: (n_items, n_bins)
        self.edges: NDArray[np.float64] | None = None
        self.scale: tuple[float, float] = (1.0, 5.0)

    def fit(self, train: pd.DataFrame) -> Self:
        r = train["rating"].to_numpy(dtype=float)
        u, self.users = pd.factorize(train["user"])
        i, self.items = pd.factorize(train["item"])
        self.edges = time_bin_edges(train["timestamp"], self.n_bins)
        # np.unique may drop duplicate edges
        nb = len(self.edges) - 1
        # one code per (item, bin)
        ib = i * nb + time_bins(train["timestamp"], self.edges)

        n_users, n_items = len(self.users), len(self.items)
        n_u = np.bincount(u, minlength=n_users)
        n_i = np.bincount(i, minlength=n_items)
        n_ib = np.bincount(ib, minlength=n_items * nb)

        alpha = r.mean()
        b_u, b_i, b_ib = np.zeros(n_users), np.zeros(
            n_items), np.zeros(n_items * nb)
        for _ in range(self.n_iters):
            alpha = np.mean(r - b_u[u] - b_i[i] - b_ib[ib])
            b_u = np.bincount(
                u, weights=r - alpha - b_i[i] - b_ib[ib], minlength=n_users) / (self.reg_u + n_u)
            b_i = np.bincount(
                i, weights=r - alpha - b_u[u] - b_ib[ib], minlength=n_items) / (self.reg_i + n_i)
            b_ib = np.bincount(
                ib, weights=r - alpha - b_u[u] - b_i[i], minlength=n_items * nb) / (self.reg_ib + n_ib)

        self.alpha, self.b_u, self.b_i = float(alpha), b_u, b_i
        self.b_ib = b_ib.reshape(n_items, nb)
        self.scale = (float(r.min()), float(r.max()))

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray:
        if self.users is None:
            raise RuntimeError("TimeBiasModel.predict called before fit")
        up = self.users.get_indexer(rows["user"])              # -1 if unseen
        ip = self.items.get_indexer(rows["item"])
        bins = time_bins(rows["timestamp"], self.edges)

        pred = np.full(len(up), self.alpha)
        pred += np.where(up >= 0, self.b_u[up], 0.0)
        pred += np.where(ip >= 0, self.b_i[ip], 0.0)
        pred += np.where(ip >= 0, self.b_ib[ip, bins], 0.0)

        return pred.clip(*self.scale)
