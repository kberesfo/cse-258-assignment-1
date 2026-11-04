import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import clip_timestamps

_TIME_PARTS = {
    "weekday": lambda t: t.dt.weekday,
    "month": lambda t: t.dt.month,
    "year": lambda t: t.dt.year,
    "hour": lambda t: t.dt.hour,
}


def time_onehot(
    ts: pd.Series,
    parts: tuple[str, ...] = ("weekday", "month", "year"),
    columns: pd.Index | None = None,
) -> pd.DataFrame:
    """
    One-hot calendar features from Unix timestamps (clipped to the plausible range first).

    Training data: call without `columns`; one level per part is dropped (avoids redundancy).
    Validation/test: pass the training result's .columns so every set has identical columns
    in the same order (unseen categories are dropped, missing ones are all-zero).
    """
    t = pd.to_datetime(clip_timestamps(ts), unit="s")
    raw = pd.DataFrame(
        {p: _TIME_PARTS[p](t) for p in parts},
        index=ts.index
    ).astype(str)

    out = pd.get_dummies(raw, drop_first=columns is None, dtype=float)

    if columns is not None:
        out = out.reindex(columns=columns, fill_value=0.0)

    return out


def time_bin_edges(ts: pd.Series, n_bins: int = 20) -> NDArray[np.float64]:
    """Bin edges from TRAINING timestamps (clipped), using quantiles so each bin holds ~equal rows."""
    levels = np.linspace(
        0, 1, n_bins + 1)                  # 0, 1/n, 2/n, ..., 1
    edges = np.quantile(clip_timestamps(ts).to_numpy(), levels)
    # drop duplicate edges (identical timestamps)

    return np.unique(edges)


def time_bins(ts: pd.Series, edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Bin index 0..n_bins-1 for each timestamp (clipped first), using edges from training data."""
    values = clip_timestamps(ts).to_numpy()

    return np.searchsorted(edges[1:-1], values, side="right")
