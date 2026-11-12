"""Time-local item similarity (Task 4): how similar is a candidate to what the user was reading
around the query time t? Global similarity features average over the whole history; this one
looks only at the user's interactions nearest to t (before and after), which catches series
and binge reading."""

from typing import Self

import numpy as np
import pandas as pd

from assignment_1.modules.util import clip_timestamps

from ..similarity import ItemSimilarity, Similarity, binary_cosine


class LocalSimilarity:
    """
    For each (user, item, t): similarity of item to the user's 2·n_side training interactions
    nearest in time to t (n_side before, n_side after), summarised as
      loc_w{w}_sum / _max   over neighbours within ±w days
      loc_k{k}_mean / _max  over the k neighbours nearest in time
      loc_decay             Σ s · exp(−|Δt| / τ days)
    Unknown user or item → all zeros.
    """

    def __init__(self, similarity: Similarity = binary_cosine, n_side: int = 10,
                 windows: tuple[int, ...] = (1, 7, 30), k_near: tuple[int, ...] = (1, 3, 10),
                 tau: float = 7.0) -> None:
        self.similarity, self.n_side = similarity, n_side
        self.windows, self.k_near, self.tau = windows, k_near, tau

    def fit(self, train: pd.DataFrame, sim: ItemSimilarity | None = None) -> Self:
        """sim: an ItemSimilarity already fitted on the same train (reused instead of refit)."""
        self.sim = sim if sim is not None else ItemSimilarity(self.similarity).fit(train)
        self.S = self.sim.S.tocsr()
        u = self.sim.users.get_indexer(train["user"])
        i = self.sim.items.get_indexer(train["item"])
        t = clip_timestamps(train["timestamp"]).to_numpy(float)
        order = np.lexsort((t, u))                       # by user, then time
        self.u_s, self.i_s, self.t_s = u[order], i[order], t[order]
        # one sorted key so a single searchsorted finds t within each user's history;
        # user * 1e10 + t stays exact in float64 while users < ~9e5
        assert len(self.sim.users) < 9e5
        self.key = self.u_s * 1e10 + self.t_s
        self.start = np.searchsorted(self.u_s, np.arange(len(self.sim.users) + 1))
        return self

    def features(self, users: pd.Series, items: pd.Series, ts: pd.Series) -> pd.DataFrame:
        u = self.sim.users.get_indexer(users)
        i = self.sim.items.get_indexer(items)
        t = clip_timestamps(ts).to_numpy(float)
        known = (u >= 0) & (i >= 0)
        u0, i0 = np.where(known, u, 0), np.where(known, i, 0)

        # neighbours in time: n_side positions either side of t in the user's sorted history
        p = np.searchsorted(self.key, u0 * 1e10 + t)
        idx = p[:, None] + np.arange(-self.n_side, self.n_side)
        valid = ((idx >= self.start[u0][:, None]) & (idx < self.start[u0 + 1][:, None])
                 & known[:, None])
        idx = np.clip(idx, 0, len(self.i_s) - 1)
        dt = np.where(valid, np.abs(self.t_s[idx] - t[:, None]) / 86400, np.inf)
        h = self.i_s[idx]
        s = np.asarray(self.S[np.repeat(i0, idx.shape[1]), h.ravel()]).reshape(idx.shape)
        s = np.where(valid, s, 0.0)

        out = {}
        for w in self.windows:
            sw = s * (dt <= w)
            out[f"loc_w{w}_sum"] = sw.sum(1)
            out[f"loc_w{w}_max"] = sw.max(1)
        s_near = np.take_along_axis(s, np.argsort(dt, axis=1, kind="stable"), 1)
        n_valid = valid.sum(1)
        for k in self.k_near:
            out[f"loc_k{k}_mean"] = s_near[:, :k].sum(1) / np.clip(np.minimum(n_valid, k), 1, None)
            out[f"loc_k{k}_max"] = s_near[:, :k].max(1)
        out["loc_decay"] = (s * np.exp(-dt / self.tau)).sum(1)
        return pd.DataFrame(out)
