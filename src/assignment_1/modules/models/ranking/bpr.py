"""Bayesian Personalized Ranking (PML §5.2.2) via implicit: ranks items a user interacted with
above items they didn't. Scores are unnormalized (item bias + γ_u · γ_i), used for ranking only."""

from typing import Any, Self

import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix


class BPRModel:
    """BPR fitted on every (user, item) row; score(users, items) gives one score per row."""

    def __init__(self, seed: int = 0, **bpr_params: Any) -> None:
        """bpr_params: factors, regularization, learning_rate, iterations (implicit kwargs)."""
        self.seed = seed
        self.bpr_params = bpr_params
        self.model: BayesianPersonalizedRanking | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        self.user_index = {u: k for k, u in enumerate(train["user"].unique())}
        self.item_index = {i: k for k, i in enumerate(train["item"].unique())}
        rows = train["user"].map(self.user_index).to_numpy()
        cols = train["item"].map(self.item_index).to_numpy()

        X = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)),
                       shape=(len(self.user_index), len(self.item_index)))
        # repeat interactions count once
        X.data[:] = 1

        # num_threads=1 + random_state: deterministic fits, so make check reproduces dist/
        self.model = BayesianPersonalizedRanking(
            random_state=self.seed,
            num_threads=1,
            **self.bpr_params
        )

        self.model.fit(X, show_progress=False)

        return self

    def score(self, users: ArrayLike, items: ArrayLike) -> NDArray[np.float64]:
        """Row-wise γ_u · γ_i (implicit stores the item bias as an extra factor); 0 if unknown."""
        if self.model is None:
            raise RuntimeError("BPRModel.score called before fit")
        u_idx = pd.Series(users).map(self.user_index)
        i_idx = pd.Series(items).map(self.item_index)
        known = (u_idx.notna() & i_idx.notna()).to_numpy()

        scores = np.zeros(len(u_idx))
        U = self.model.user_factors[u_idx[known].astype(int).to_numpy()]
        V = self.model.item_factors[i_idx[known].astype(int).to_numpy()]
        scores[known] = (U * V).sum(axis=1)
        return scores

    # a fitted model *is* a scorer: calc_pair_accuracy(pairs, bpr)
    __call__ = score
