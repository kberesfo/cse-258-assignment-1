from typing import Self

import numpy as np
import pandas as pd

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
