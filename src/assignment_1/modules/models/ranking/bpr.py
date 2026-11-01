from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix


@dataclass
class BPRFit:
    model: BayesianPersonalizedRanking
    user_index: dict[str, int]  # user ID -> row in user_factors
    item_index: dict[str, int]  # item ID -> row in item_factors


def fit_bpr(train_df: pd.DataFrame, seed: int = 0, **bpr_params: Any) -> BPRFit:
    """Train implicit BPR on every (user, item) row of train_df."""
    # 1. build user_index / item_index from train_df
    # 2. build the sparse users x items matrix of 1s (csr_matrix((data, (rows, cols)), shape=...))
    # 3. model = BayesianPersonalizedRanking(random_state=seed, num_threads=1, **bpr_params); model.fit(X)
    user_index = {u: k for k, u in enumerate(train_df["user"].unique())}
    item_index = {i: k for k, i in enumerate(train_df["item"].unique())}
    # each interaction's user -> row number
    rows = train_df["user"].map(user_index).to_numpy()
    # each interaction's item -> column number
    cols = train_df["item"].map(item_index).to_numpy()
    data = np.ones(len(train_df), dtype=np.float32)  # a 1 at each (row, col)

    X = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(item_index)))
    X.data[:] = 1

    model = BayesianPersonalizedRanking(random_state=seed, num_threads=1, **bpr_params)
    model.fit(X, show_progress=False)

    return BPRFit(model, user_index, item_index)


def bpr_scorer(
    fit: BPRFit,
) -> Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]:
    """Return score_fn(users, items) -> one BPR score per row, for calc_pair_accuracy."""

    def score(users: NDArray, items: NDArray) -> NDArray[np.float64]:
        # map IDs -> indices, gather factor rows, row-wise dot product
        u_idx = pd.Series(users).map(fit.user_index)
        i_idx = pd.Series(items).map(fit.item_index)

        known = (u_idx.notna() & i_idx.notna()).to_numpy()

        scores = np.zeros(len(users))

        u_idx, i_idx = (
            u_idx[known].astype(int).to_numpy(),
            i_idx[known].astype(int).to_numpy(),
        )

        U = fit.model.user_factors[u_idx]
        V = fit.model.item_factors[i_idx]
        # row wise dot
        scores[known] = (U * V).sum(axis=1)

        return scores

    return score
