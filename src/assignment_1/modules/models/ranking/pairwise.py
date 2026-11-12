"""Pairwise candidate features for the ranking tasks (2/4/5): φ(u, item1) − φ(u, item2).

Every statistic is fitted on one training set (tr for validation, all of train for the test).
Query time is the row's timestamp when pairs have one (Task 4), else the user's last training
interaction (Task 5).
"""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from assignment_1.modules.util import clip_timestamps, time_bin_edges, time_bins

from ..similarity import ItemSimilarity, Similarity, binary_cosine
from .bpr import BPRModel
from .local import LocalSimilarity
from .markov import TransitionModel


class PairFeatures:
    """
    Candidate features (all from training history only):
      bpr, not_yet, log_age, log_pop, trend_{nb} for each nb in trend_bins
      + trans, unseen                      if transitions=True (Task 5)
      + {sim_prefix}_sum / _max / _mean    if similarity is given
      + loc_*                              if local_similarity=True (Task 4; see LocalSimilarity)
    """

    def __init__(
        self,
        bpr_params: dict,
        trend_bins: tuple[int, ...] = (20, 50, 100),
        transitions: bool = False,
        similarity: Similarity | None = None,
        sim_prefix: str = "cos",
        local_similarity: bool = False,
    ) -> None:
        self.bpr_params = bpr_params
        self.trend_bins = trend_bins
        self.transitions = transitions
        self.similarity = similarity
        self.sim_prefix = sim_prefix
        self.local_similarity = local_similarity
        self.bpr: BPRModel | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        ts = clip_timestamps(train["timestamp"])
        self.bpr = BPRModel(**self.bpr_params).fit(train)
        # item -> first interaction time
        self.first_seen = ts.groupby(train["item"]).min()
        # item -> interaction count
        self.item_tot = train["item"].value_counts()
        # user -> last interaction time
        self.t_last = train.groupby("user")["timestamp"].max()
        # fallback query time
        self.t_max = float(train["timestamp"].max())

        # nb -> (edges, (item, bin) counts)
        self.trend_tables = {}
        for nb in self.trend_bins:
            edges = time_bin_edges(train["timestamp"], nb)
            bins = time_bins(train["timestamp"], edges)
            self.trend_tables[nb] = (
                edges, train.groupby([train["item"], bins]).size())

        self.trans = TransitionModel().fit(train) if self.transitions else None
        self.sim = (ItemSimilarity(self.similarity, prefix=self.sim_prefix).fit(train)
                    if self.similarity is not None else None)
        # reuses the fitted global similarity when it is the same measure
        self.local = (LocalSimilarity(self.similarity or binary_cosine).fit(train, sim=self.sim)
                      if self.local_similarity else None)
        return self

    def query_times(self, pairs: pd.DataFrame) -> pd.Series:
        """Row timestamp when pairs have one (Task 4); else the user's last training time (Task 5)."""
        if "timestamp" in pairs:
            return pairs["timestamp"]
        return pairs["user"].map(self.t_last).fillna(self.t_max)

    def candidate(self, users: pd.Series, items: pd.Series, ts: pd.Series) -> pd.DataFrame:
        """Features of one candidate per row, at query times ts."""
        if self.bpr is None:
            raise RuntimeError("PairFeatures.candidate called before fit")
        t = clip_timestamps(ts).to_numpy(dtype=float)
        fs = items.map(self.first_seen).to_numpy(dtype=float)
        tot = items.map(self.item_tot).fillna(0).to_numpy(dtype=float)
        age = t - fs

        feats = {
            "bpr": self.bpr(users.to_numpy(), items.to_numpy()),
            "not_yet": (age < 0).astype(float),
            "log_age": np.log1p(np.clip(np.nan_to_num(age, nan=0.0), 0, None) / 86400),
            "log_pop": np.log1p(tot),
        }
        for nb, (edges, tbl) in self.trend_tables.items():
            idx = pd.MultiIndex.from_arrays([items, time_bins(ts, edges)])
            feats[f"trend_{nb}"] = tbl.reindex(
                idx, fill_value=0).to_numpy(dtype=float) / (tot + 1)

        if self.trans is not None:
            feats["trans"] = np.log1p(self.trans(
                users.to_numpy(), items.to_numpy()))
            feats["unseen"] = (tot == 0).astype(float)

        F = pd.DataFrame(feats)
        if self.sim is not None:
            F = pd.concat([F, self.sim.features(users, items)], axis=1)
        if self.local is not None:
            F = pd.concat([F, self.local.features(users, items, ts)], axis=1)
        return F

    def transform(self, pairs: pd.DataFrame) -> NDArray:
        """φ(u, item1) − φ(u, item2) for each pair; positive values favour item1."""
        users, ts = pairs["user"], self.query_times(pairs)
        return (self.candidate(users, pairs["item1"], ts)
                - self.candidate(users, pairs["item2"], ts)).to_numpy()
