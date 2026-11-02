from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass
class TransitionFit:
    counts: pd.Series      # (prev item, next item) -> number of times observed
    last_item: pd.Series   # user -> their most recent item in the training data


def fit_transitions(train_df: pd.DataFrame) -> TransitionFit:
    """
    Count item-to-item transitions in each user's time-ordered history.

    Returns TransitionFit: counts[(prev, next)] = times `next` directly followed `prev`;
    last_item[user] = the user's most recent item in train_df.
    """
    sorted_df = train_df.sort_values(["user", "timestamp"], kind="stable")
    nxt = sorted_df.groupby("user")["item"].shift(-1)
    trans = pd.DataFrame({"prev": sorted_df["item"], "next": nxt}).dropna()

    # one fewer transition than rows, per user
    assert len(trans) == len(train_df) - train_df["user"].nunique()
    counts = trans.groupby(['prev', "next"]).size()
    last_item = sorted_df.groupby("user")["item"].last()

    return TransitionFit(counts, last_item)


def transition_scorer(fit: TransitionFit) -> Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]:
    """Return score_fn(users, items) -> #(user's last item -> item) transitions, 0 if never seen."""

    def score(users: ArrayLike, items: ArrayLike) -> NDArray[np.float64]:
        last = pd.Series(users).map(fit.last_item)
        
        keys = pd.MultiIndex.from_arrays([last, np.asarray(items)])
        scores = fit.counts.reindex(keys, fill_value=0).to_numpy(dtype=float)

        return scores
    
    return score
