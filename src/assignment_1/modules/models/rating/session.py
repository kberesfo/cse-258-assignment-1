from typing import Self

import numpy as np
import pandas as pd

from assignment_1.modules.util import clip_timestamps

from .base import RatingModel
from .stacking import FeatureSource


class SessionResidual(FeatureSource):
    """Shrunk mean residual (r − b̂) of the user's training rows near the row's timestamp."""

    def __init__(self, windows=(("day", 86400), ("week", 7 * 86400)), shrink: float = 1.0) -> None:
        self.windows, self.shrink = windows, shrink

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        res = train["rating"].to_numpy() - base.predict(train)
        self.tables = {}
        for name, w in self.windows:
            key = pd.MultiIndex.from_arrays(
                [train["user"], train["timestamp"] // w])
            self.tables[name] = (w, pd.Series(res, index=key).groupby(
                level=[0, 1]).agg(["sum", "size"]))
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for name, (w, tbl) in self.tables.items():
            s = tbl.reindex(pd.MultiIndex.from_arrays(
                [rows["user"], rows["timestamp"] // w]))
            out[f"sess_{name}"] = (
                s["sum"] / (s["size"] + self.shrink)).fillna(0).to_numpy()
            out[f"log_n_{name}"] = np.log1p(s["size"].fillna(0).to_numpy())
        return pd.DataFrame(out)


class SequenceResidual(FeatureSource):
    """Mean residual of the user's k training rows just before / after the row's timestamp, and
    the days to the nearest one on each side. Follows the user's rating mood across sparse
    periods, where SessionResidual's fixed day / week windows are empty."""

    def __init__(self, ks: tuple[int, ...] = (1, 3, 5)) -> None:
        self.ks = ks

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        u, self.users = pd.factorize(train["user"])
        ts = clip_timestamps(train["timestamp"]).to_numpy(dtype=float)
        key = u.astype(np.float64) * 2e9 + ts             # sorts by user, then time
        order = np.argsort(key, kind="stable")
        res = train["rating"].to_numpy(dtype=float) - base.predict(train)
        self.key_, self.user_, self.ts_ = key[order], u[order], ts[order]
        self.cum_ = np.r_[0, np.cumsum(res[order])]
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        u = self.users.get_indexer(rows["user"])
        q = clip_timestamps(rows["timestamp"]).to_numpy(dtype=float)
        pos = np.searchsorted(self.key_, u * 2e9 + q, side="left")
        last = len(self.key_) - 1
        out = {}
        for k in self.ks:
            for side, lo, hi in (("b", np.maximum(pos - k, 0), pos),
                                 ("a", pos, np.minimum(pos + k, last + 1))):
                n = hi - lo
                same = ((n > 0) & (u >= 0) & (self.user_[np.clip(lo, 0, last)] == u)
                        & (self.user_[np.clip(hi - 1, 0, last)] == u))
                mean = (self.cum_[hi] - self.cum_[lo]) / np.maximum(n, 1)
                out[f"seq_{side}{k}"] = np.where(same, mean, np.nan)
        before, after = np.clip(pos - 1, 0, last), np.clip(pos, 0, last)
        out["seq_dt_b"] = np.where((self.user_[before] == u) & (u >= 0) & (pos > 0),
                                   (q - self.ts_[before]) / 86400, np.nan)
        out["seq_dt_a"] = np.where((self.user_[after] == u) & (u >= 0) & (pos <= last),
                                   (self.ts_[after] - q) / 86400, np.nan)
        return pd.DataFrame(out)
