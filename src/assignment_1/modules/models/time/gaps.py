"""Time until a user's next interaction (Task 6, scored by MAE).

Base prediction: per-user median of the last k gaps, shrunk toward the global median in log space
(MAE is minimized by the median, so medians rather than means; formula sheet 6.3):
    log ĝ_u = w_u log m_u + (1 − w_u) log m,   w_u = n_u / (n_u + λ)
Optional lower bound: the next interaction can't come before the candidate item first appears in
training, so ĝ ≥ first_seen(item) − t_last(user) + offset.
"""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class GapModel:
    def __init__(
        self,
        k: int | None = 10,
        lam: float = 1.5,
        bound_offset: float | None = None
    ) -> None:
        """k: use each user's last k gaps (None = all); lam: shrinkage toward the global median;
        bound_offset: seconds added to the item first-seen bound (None = no bound)."""
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
