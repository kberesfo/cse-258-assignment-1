"""Item-kNN on a base model's residuals (PML §4.3, similarity-based rating prediction)."""

from typing import Self

import numpy as np
import pandas as pd

from ..similarity import ItemSimilarity, Similarity, binary_cosine
from .base import RatingModel
from .stacking import FeatureSource


class ItemKNN(FeatureSource):
    """
    Residual kNN features relative to a fitted base model.

    For user u and item i, with history H_u and item-item similarity S(i, j):
        num = Σ_{j ∈ H_u} S(i, j) · (r_uj − b̂_uj)      (similarity-weighted residuals)
        den = Σ_{j ∈ H_u} |S(i, j)|                     (similarity support)
    Features: {prefix} = num / (den + shrink), {prefix}_log_support = log(1 + den).
    `similarity` may be any (R, Rres) -> S function, e.g. binary_cosine, jaccard, adjusted_cosine.
    """

    def __init__(self, similarity: Similarity = binary_cosine, shrink: float = 1.0,
                 prefix: str = "knn", chunk: int = 50_000) -> None:
        self.similarity, self.shrink, self.prefix, self.chunk = similarity, shrink, prefix, chunk
        self.sim: ItemSimilarity | None = None

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        """train: rated rows the base model was fit on; base: a fitted RatingModel."""
        res = train["rating"].to_numpy() - base.predict(train)
        self.sim = ItemSimilarity(self.similarity).fit(train, residuals=res)
        return self

    def parts(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(num, den) per row; 0 where the user or item is unknown."""
        if self.sim is None:
            raise RuntimeError("ItemKNN.parts called before fit")
        sim = self.sim
        u = sim.users.get_indexer(rows["user"])
        i = sim.items.get_indexer(rows["item"])
        num, den = np.zeros(len(u)), np.zeros(len(u))

        for s in range(0, len(u), self.chunk):
            uu, ii = u[s:s + self.chunk], i[s:s + self.chunk]
            ok = (uu >= 0) & (ii >= 0)
            Si = sim.S[ii[ok]]
            num[s:s + self.chunk][ok] = np.asarray(
                sim.Rres[uu[ok]].multiply(Si).sum(axis=1)).ravel()
            den[s:s + self.chunk][ok] = np.asarray(
                abs(sim.R[uu[ok]].multiply(Si)).sum(axis=1)).ravel()
        return num, den

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        num, den = self.parts(rows)
        p = self.prefix
        return pd.DataFrame({p: num / (den + self.shrink), f"{p}_log_support": np.log1p(den)})
