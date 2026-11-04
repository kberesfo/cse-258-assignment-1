from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass
class GapFit:
    per_user: pd.Series      # user -> predicted gap (seconds)
    global_median: float     # fallback for users with no gaps


def fit_gap_model(train_df: pd.DataFrame, k: int | None = 10, lam: float = 1.5) -> GapFit:
    """Per-user median of each user's last k gaps, shrunk toward the global median in log space."""

    s = train_df.sort_values(["user", "timestamp"], kind="stable")
    gaps = s.assign(gap=s.groupby("user")[
                    "timestamp"].diff()).dropna(subset=["gap"])

    # all gaps (matches what was validated)
    m_global = float(gaps["gap"].median())
    if k is not None:
        gaps = gaps.groupby("user").tail(k)

    g = gaps.groupby("user")["gap"]
    m_u, n_u = g.median(), g.size()

    # one weight per user, in [0, 1)
    w = n_u / (n_u + lam)
    # one blended log-prediction per user
    log_pred = w * np.log(m_u) + (1 - w) * np.log(m_global)

    pred = np.exp(log_pred)
    return GapFit(per_user=pred, global_median=m_global)


def predict_gaps(fit: GapFit, users: ArrayLike) -> NDArray[np.float64]:
    """Predicted time (seconds) until each user's next interaction; unknown users -> global median."""
    return pd.Series(users).map(fit.per_user).fillna(fit.global_median).to_numpy()


def item_first_seen_bound(
    train_df: pd.DataFrame, users: ArrayLike, items: ArrayLike
) -> NDArray[np.float64]:
    """Lower bound on each user's next gap: time from the user's last interaction in train_df
    until the item first appears in train_df. NaN where the item never appears or the user is unknown."""
    first_seen = train_df.groupby("item")["timestamp"].min()
    t_last = train_df.groupby("user")["timestamp"].max()

    lo = pd.Series(items).map(first_seen).to_numpy() - \
        pd.Series(users).map(t_last).to_numpy()

    return lo.astype(float)
