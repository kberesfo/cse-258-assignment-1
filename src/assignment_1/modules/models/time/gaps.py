"""Time until a user's next interaction (Task 6, scored by MAE).

MAE is minimized by the median, so every model here estimates a median gap, not a mean.

GapModel: per-user median of the last k gaps, shrunk toward the global median in log space
(formula sheet 6.3):
    log ĝ_u = w_u log m_u + (1 − w_u) log m,   w_u = n_u / (n_u + λ)
Optional lower bound: the next interaction can't come before the candidate item first appears in
training, so ĝ ≥ first_seen(item) − t_last(user) + offset.

GapRegressor: time-series regression on the user's previous gaps (lecture hint: predict
delta_{n+1} from delta_1..n). A gradient-boosted regressor with MAE loss on log1p(gap); because
median(log g) = log(median g), fitting MAE in log space still estimates the median. Features:
the last n_lags log gaps, log-gap median / mean / std / count, GapModel's prediction, the
first-seen bound and item popularity. Trained by stepping back one interaction (each user's last
training gap is the target); the bound is applied to its output as a hard constraint.

All times are in seconds (SECONDS_PER_DAY = 86,400).
"""
import copy
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

SECONDS_PER_DAY = 86_400


def split_last(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(history without each user's last interaction, that last row with its gap as `target`).
    Users whose last row has no previous interaction (a single row) are dropped from `last`."""
    s = df.sort_values(["user", "timestamp"], kind="stable")
    s = s.assign(gap=s.groupby("user")["timestamp"].diff())
    is_last = (s.groupby("user").cumcount(ascending=False) == 0).to_numpy()
    last = s[is_last & s["gap"].notna().to_numpy()].rename(
        columns={"gap": "target"})
    return s[~is_last].drop(columns="gap"), last


class GapModel:
    def __init__(
        self,
        k: int | None = 10,
        lam: float = 1.5,
        bound_offset: float | None = None
    ) -> None:
        """Per-user shrunk median gap, optionally raised to the item first-seen bound."""
        self.k = k
        self.lam = lam
        self.bound_offset = bound_offset
        self.per_user: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        s = train.sort_values(["user", "timestamp"], kind="stable")
        gaps = s.assign(gap=s.groupby("user")[
                        "timestamp"].diff()).dropna(subset=["gap"])

        # all gaps (matches what was validated)
        self.global_median = float(gaps["gap"].median())
        if self.k is not None:
            gaps = gaps.groupby("user").tail(self.k)

        g = gaps.groupby("user")["gap"]
        m_u, n_u = g.median(), g.size()
        # one weight per user, in [0, 1)
        w = n_u / (n_u + self.lam)
        self.per_user = np.exp(w * np.log(m_u) + (1 - w)
                               * np.log(self.global_median))

        # for the bound
        self.first_seen = train.groupby("item")["timestamp"].min()
        self.t_last = train.groupby("user")["timestamp"].max()
        return self

    def bound(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """first_seen(item) − t_last(user); NaN where the item or user is unknown."""
        return (rows["item"].map(self.first_seen).to_numpy(dtype=float)
                - rows["user"].map(self.t_last).to_numpy(dtype=float))

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        """Predicted gap (seconds) per row; unknown users get the global median."""
        if self.per_user is None:
            raise RuntimeError("GapModel.predict called before fit")
        pred = rows["user"].map(self.per_user).fillna(
            self.global_median).to_numpy(dtype=float)
        if self.bound_offset is not None:
            # NaN bound -> keep pred
            pred = np.fmax(pred, self.bound(rows) + self.bound_offset)
        return pred


class GapRegressor:
    """
    Task 6: time-delta prediction. How long (seconds) until the user's next interaction?
    Metric: MAE (lower is better).

    Test design
    -----------
    One row per user, with the item of the next interaction. Metric confirmed as MAE; gaps are
    heavily skewed (median ~13 days, mean ~71 days), so medians, not means.

    Model
    -----
    GapRegressor (time-series regression, as suggested in lecture): HistGradientBoostingRegressor
    with MAE loss on log1p(gap), trained by stepping back one interaction per user. Features: the
    last 5 log gaps, log-gap median / mean / std / count and median of the last 10, the GapModel
    prediction (per-user median of the last 10 gaps, shrunk toward the global median in log space,
    λ = 2), the item first-seen bound (days) and item popularity.
    Lower bound, applied to the output: the next book can't be read before it first appears in
    training, so ĝ ≥ first_seen(item) − t_last(user) + 35 days (the median delay after a book first
    appears). An upper bound from the end of the data hurt (corrupt future timestamps), not used.

    Validation
    ----------
    Each user's last training gap held out, predicted from the earlier history
    (notebooks/task6.ipynb):
    per-user median 12,052,136 -> shrunk median, last 10 gaps (λ 1.5) 10,137,223
    -> + first-seen bound 8,531,601 -> + 35-day offset, λ 2: <E4>
    -> GapRegressor: <E5>
    Public: 13,013,427 -> ~11.39M (bound) -> 11,209,411 (offset, λ 2) -> <after upload>.
    """

    def __init__(self, base: GapModel | None = None, n_lags: int = 5,
                 bound_offset: float = 35 * SECONDS_PER_DAY, regressor=None) -> None:
        self.base = base if base is not None else GapModel(k=10, lam=2.0)
        self.n_lags, self.bound_offset = n_lags, bound_offset
        self.regressor = (regressor if regressor is not None
                          else HistGradientBoostingRegressor(loss="absolute_error", random_state=0))
        self.reg_ = None

    def features(self, hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-row features from `hist` (the user's previous gaps, the base model, the bound)."""
        s = hist.sort_values(["user", "timestamp"], kind="stable")
        s = s.assign(lg=np.log1p(s.groupby("user")[
                     "timestamp"].diff())).dropna(subset=["lg"])
        r = s.groupby("user").cumcount(
            ascending=False)               # 0 = most recent gap
        k = self.n_lags
        lags = s[r < k].assign(r=r[r < k]).pivot(
            index="user", columns="r", values="lg")
        # always lag_0 … lag_{k−1}
        lags = lags.reindex(columns=range(k))
        lags.columns = [f"lag_{c}" for c in range(k)]
        stats = s.groupby("user")["lg"].agg(
            ["median", "mean", "std", "count"]).add_prefix("lg_")
        stats["lg_median10"] = s[r < 10].groupby("user")["lg"].median()

        u = rows["user"]
        F = pd.concat([lags.reindex(u).reset_index(drop=True),
                       stats.reindex(u).reset_index(drop=True)], axis=1)
        # no bound: it's a feature here
        base = copy.deepcopy(self.base).fit(hist)
        F["log_base"] = np.log1p(base.predict(rows))
        F["bound_days"] = base.bound(rows) / SECONDS_PER_DAY
        F["log_item_pop"] = np.log1p(rows["item"].map(hist["item"].value_counts())
                                     .fillna(0).to_numpy(dtype=float))
        return F

    def fit(self, train: pd.DataFrame) -> Self:
        hist_b, last_b = split_last(train)
        self.reg_ = clone(self.regressor).fit(self.features(hist_b, last_b),
                                              np.log1p(last_b["target"].to_numpy()))
        self.train_ = train
        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.reg_ is None:
            raise RuntimeError("GapRegressor.predict called before fit")
        F = self.features(self.train_, rows)
        pred = np.expm1(self.reg_.predict(F))
        bound = F["bound_days"].to_numpy() * SECONDS_PER_DAY
        # hard constraint; NaN -> keep pred
        return np.fmax(pred, bound + self.bound_offset)
