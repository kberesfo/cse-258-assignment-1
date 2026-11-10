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
the user's previous gaps and recent activity, GapModel's prediction, and when other readers read
the next book relative to the user's last interaction. Trained by stepping back one interaction
(each user's last training gap is the target).

All times are in seconds (SECONDS_PER_DAY = 86,400).
"""
import copy
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.util import TS_MAX, TS_MIN

SECONDS_PER_DAY = 86_400
# Just after the last clean timestamp (2017-11-01). Fixed, so the feature means the same on the
# stepped-back history as on the full training set.
GAP_DATA_END = pd.Timestamp("2017-11-05").timestamp()
GAP_ITEM_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
GAP_WINDOWS_DAYS = (30, 90, 365)


def _clean_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with plausible timestamps; corrupt ones would swamp the time statistics."""
    return df[df["timestamp"].between(TS_MIN, TS_MAX)]


def _item_time_features(hist: pd.DataFrame, rows: pd.DataFrame, t: NDArray) -> pd.DataFrame:
    """When the row's item was read by others, relative to the user's last interaction t (days).

    The next interaction is after t, so the item's readers after t are the most informative.
    NaN where the item is unknown (or, for the quantiles, has no reader after t).
    """
    by_item = {k: np.sort(v.to_numpy())
               for k, v in _clean_timestamps(hist).groupby("item")["timestamp"]}
    n_q, n_w = len(GAP_ITEM_QUANTILES), len(GAP_WINDOWS_DAYS)
    out = np.full((len(rows), 6 + n_q + n_w), np.nan)
    for n, (item, t_n) in enumerate(zip(rows["item"].to_numpy(), t, strict=True)):
        times = by_item.get(item)
        if times is None:
            continue
        j = np.searchsorted(times, t_n, side="right")
        after = times[j:]
        out[n, :6] = (len(times), len(after), len(after) / len(times),
                      (times[0] - t_n) / SECONDS_PER_DAY, (times[-1] - t_n) / SECONDS_PER_DAY, j)
        if len(after):
            out[n, 6:6 + n_q] = (np.quantile(after, GAP_ITEM_QUANTILES) - t_n) / SECONDS_PER_DAY
        for m, w in enumerate(GAP_WINDOWS_DAYS):
            out[n, 6 + n_q + m] = np.searchsorted(after, t_n + w * SECONDS_PER_DAY, side="right")

    F = pd.DataFrame(out, columns=[
        "it_n", "it_after", "it_frac_after", "it_first", "it_last", "it_before",
        *[f"it_q{int(q * 100)}" for q in GAP_ITEM_QUANTILES],
        *[f"it_{w}" for w in GAP_WINDOWS_DAYS],
    ])
    counts = ["it_n", "it_after", "it_before", *[f"it_{w}" for w in GAP_WINDOWS_DAYS]]
    F[counts] = np.log1p(F[counts])
    return F


def _coread_features(hist: pd.DataFrame, rows: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Readers of the row's item who also read one of the user's last k items (plan_t6.md P3).

    For a co-reader v of history item a: the user read a at t_ua, v read the row's item
    t_vb − t_va after a, so the implied next time is t_ua + (t_vb − t_va). Features, pooled over
    the last k items: how many co-readers, the share whose implied time is after the user's last
    interaction t, and quartiles of (implied − t) in log days. With k = 1, a is the last item and
    the implied gap is just v's delay. NaN where nobody read both."""
    s = (_clean_timestamps(hist).sort_values(["user", "timestamp"], kind="stable")
         .drop_duplicates(["user", "item"], keep="last"))
    recent = s.groupby("user").tail(k)
    recent_items = {u: list(zip(v["item"].to_numpy(), v["timestamp"].to_numpy(), strict=True))
                    for u, v in recent.groupby("user")[["item", "timestamp"]]}
    t_last = s.groupby("user")["timestamp"].last()
    readers = {i: dict(zip(v["user"].to_numpy(), v["timestamp"].to_numpy(), strict=True))
               for i, v in s.groupby("item")[["user", "timestamp"]]}

    out = np.full((len(rows), 5), np.nan)
    users, items = rows["user"].to_numpy(), rows["item"].to_numpy()
    for n, (user, b_item) in enumerate(zip(users, items, strict=True)):
        b = readers.get(b_item)
        hist_items = recent_items.get(user)
        if b is None or hist_items is None:
            continue
        t_u = t_last[user]
        implied = []
        for a_item, t_ua in hist_items:
            a = readers[a_item]
            small, large, sign = (a, b, 1.0) if len(a) <= len(b) else (b, a, -1.0)
            # d = t_vb − t_va for each co-reader v
            implied.extend(t_ua - t_u + sign * (large[v] - t)
                           for v, t in small.items() if v in large and v != user)
        d = np.asarray(implied, dtype=float)
        out[n, 0] = np.log1p(len(d))
        if not len(d):
            continue
        after = d[d > 0] / SECONDS_PER_DAY
        out[n, 1] = len(after) / len(d)
        if len(after):
            out[n, 2:] = np.log1p(np.quantile(after, (0.25, 0.5, 0.75)))
    return pd.DataFrame(out, columns=["co_n", "co_frac_after", "co_q25", "co_med", "co_q75"])


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
    with MAE loss on log1p(gap), trained by stepping back one interaction per user. Features:
    the user's last 8 log gaps and log-gap statistics, recent activity and calendar position, the
    GapModel prediction, and the next book's interaction times relative to the user's last
    interaction t (first / last reader, share and count of readers after t, quantiles of the
    reader times after t, readers in the next 30 / 90 / 365 days); optionally the co-readers'
    implied delay to the next book from the user's last k books (coread).
    These make the old hard lower bound (first_seen + 35 days) unnecessary; it now costs accuracy.

    Validation
    ----------
    Each user's last training gap held out, predicted from the earlier history:
    per-user median 12,052,136 -> shrunk median, last 10 gaps (λ 1.5) 10,137,223
    -> + first-seen bound 8,531,601 -> first GapRegressor 8,012,169
    -> + item reader times 7,183,000 (one step further back: 8,009,000 -> 7,191,000)
    -> + co-read delay, k = 5, 5 seeds 7,110,402 (one step further back 7,096,961).
    """

    def __init__(self, base: GapModel | None = None, n_lags: int = 8,
                 bound_offset: float | None = None, regressor=None,
                 coread: bool = False, coread_k: int = 1, n_bag: int = 1,
                 scale: float = 1.0) -> None:
        """bound_offset: apply ĝ ≥ first_seen − t_last + offset to the output; None for no bound.
        Options from plan_t6.md (defaults reproduce the shipped model):
          coread:    co-readers' implied delay to the row's item (_coread_features)
          coread_k:  pool that delay over the user's last k items
          n_bag:     regressors with different seeds, log predictions averaged
          scale:     multiply predictions by this. The model learns from the gap one step
                     before the one it predicts, and held-out gaps grow step by step (median
                     43.9 → 49.1 days from backtest B to A), so it predicts too short."""
        self.base = base if base is not None else GapModel(k=10, lam=1.5)
        self.n_lags, self.bound_offset = n_lags, bound_offset
        self.coread, self.coread_k, self.n_bag = coread, coread_k, n_bag
        self.scale = scale
        self.regressor = (regressor if regressor is not None
                          else HistGradientBoostingRegressor(
                              loss="absolute_error", learning_rate=0.03, max_iter=500,
                              min_samples_leaf=40, random_state=0))
        self.reg_ = None

    def _user_features(self, hist: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, NDArray]:
        """(the user's gap and activity features, their last clean interaction time)."""
        s = _clean_timestamps(hist).sort_values(["user", "timestamp"], kind="stable")
        s = s.assign(gap=s.groupby("user")["timestamp"].diff())
        gaps = s.dropna(subset=["gap"]).copy()
        gaps["lg"] = np.log1p(gaps["gap"].clip(lower=0))
        r = gaps.groupby("user").cumcount(ascending=False)         # 0 = most recent gap

        k = self.n_lags
        lags = gaps[r < k].assign(r=r[r < k]).pivot(index="user", columns="r", values="lg")
        lags = lags.reindex(columns=range(k))                      # always lag_0 … lag_{k−1}
        lags.columns = [f"lag_{c}" for c in range(k)]
        lg = gaps.groupby("user")["lg"]
        stats = pd.DataFrame({
            "lg_med": lg.median(), "lg_mean": lg.mean(), "lg_std": lg.std(), "lg_n": lg.size(),
            "lg_q25": lg.quantile(0.25), "lg_q75": lg.quantile(0.75), "lg_max": lg.max(),
        })
        for n in (3, 5, 10, 20):
            stats[f"lg_med{n}"] = gaps[r < n].groupby("user")["lg"].median()

        u = rows["user"]
        F = pd.concat([lags.reindex(u).reset_index(drop=True),
                       stats.reindex(u).reset_index(drop=True)], axis=1)

        by_user = s.groupby("user")["timestamp"]
        t = u.map(by_user.last()).to_numpy(dtype=float)
        when = pd.to_datetime(t, unit="s")
        F["t_last_yr"] = (t - TS_MIN) / (365.25 * SECONDS_PER_DAY)
        F["span_d"] = np.log1p((t - u.map(by_user.first()).to_numpy(dtype=float)) / SECONDS_PER_DAY)
        F["to_end_d"] = (GAP_DATA_END - t) / SECONDS_PER_DAY
        F["t_month"], F["t_dow"] = when.month, when.day_of_week

        times = {k: v.to_numpy() for k, v in by_user}
        recent = np.zeros((len(u), len(GAP_WINDOWS_DAYS)))
        for n, (user, t_n) in enumerate(zip(u.to_numpy(), t, strict=True)):
            a = times.get(user)
            if a is None or np.isnan(t_n):
                continue
            for m, w in enumerate(GAP_WINDOWS_DAYS):
                recent[n, m] = len(a) - np.searchsorted(a, t_n - w * SECONDS_PER_DAY, side="left")
        for m, w in enumerate(GAP_WINDOWS_DAYS):
            F[f"recent_{w}"] = np.log1p(recent[:, m])

        ratings = s.groupby("user")["rating"]
        F["last_rating"] = u.map(ratings.last()).to_numpy(dtype=float)
        F["mean_rating"] = u.map(ratings.mean()).to_numpy(dtype=float)
        return F, t

    def features(self, hist: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-row features from `hist`: the user's gaps and the next item's reader times."""
        rows = rows.reset_index(drop=True)
        F_user, t = self._user_features(hist, rows)
        parts = [F_user, _item_time_features(hist, rows, t)]
        if self.coread:
            parts.append(_coread_features(hist, rows, k=self.coread_k))
        F = pd.concat(parts, axis=1)
        base = copy.deepcopy(self.base).fit(hist)
        F["log_base"] = np.log1p(base.predict(rows))
        F["bound_days"] = base.bound(rows) / SECONDS_PER_DAY
        return F

    def fit(self, train: pd.DataFrame) -> Self:
        hist_b, last_b = split_last(train)
        X, y = self.features(hist_b, last_b), np.log1p(last_b["target"].to_numpy())

        self.regs_ = []
        for b in range(self.n_bag):
            reg = clone(self.regressor)
            if b and "random_state" in reg.get_params():
                reg.set_params(random_state=b)
            self.regs_.append(reg.fit(X, y))
        self.reg_ = self.regs_[0]
        self.train_ = train
        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.reg_ is None:
            raise RuntimeError("GapRegressor.predict called before fit")
        F = self.features(self.train_, rows)
        pred = self.scale * np.expm1(np.mean([reg.predict(F) for reg in self.regs_], axis=0))
        if self.bound_offset is None:
            return pred
        bound = F["bound_days"].to_numpy() * SECONDS_PER_DAY
        # hard constraint; NaN -> keep pred
        return np.fmax(pred, bound + self.bound_offset)
