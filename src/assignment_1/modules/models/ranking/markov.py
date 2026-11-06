"""First-order Markov transitions (PML §7.4): how often item j directly followed item i in users'
time-ordered histories. Scores a candidate by its count from the user's most recent item."""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


class TransitionModel:
    """counts[(prev, next)] = times `next` directly followed `prev`; last_item[user] = most recent item."""

    def __init__(self) -> None:
        self.counts: pd.Series | None = None
        self.last_item: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        ordered = train.sort_values(["user", "timestamp"], kind="stable")
        nxt = ordered.groupby("user")["item"].shift(-1)
        trans = pd.DataFrame({"prev": ordered["item"], "next": nxt}).dropna()
        # one fewer transition than rows, per user
        assert len(trans) == len(train) - train["user"].nunique()

        self.counts = trans.groupby(["prev", "next"]).size()
        self.last_item = ordered.groupby("user")["item"].last()

        return self

    def score(self, users: ArrayLike, items: ArrayLike) -> NDArray[np.float64]:
        """#(user's last item -> item) transitions; 0 if never seen or the user is unknown."""
        if self.counts is None:
            raise RuntimeError("TransitionModel.score called before fit")
        last = pd.Series(users).map(self.last_item)
        keys = pd.MultiIndex.from_arrays([last, np.asarray(items)])

        return self.counts.reindex(keys, fill_value=0).to_numpy(dtype=float)

    __call__ = score     # a fitted model is a scorer, like BPRModel
