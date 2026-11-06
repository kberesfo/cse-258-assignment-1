import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone

from assignment_1.modules.util import random_split

from .base import RatingModel


class FeatureSource(ABC):
    """Builds per-row features from training data, optionally relative to a fitted base model."""

    @abstractmethod
    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self: ...

    @abstractmethod
    def features(self, rows: pd.DataFrame) -> pd.DataFrame: ...


class RatingStats(FeatureSource):
    """User / item history statistics: how much evidence there is and how varied the user is."""

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        self.n_u = train["user"].value_counts()
        self.n_i = train["item"].value_counts()
        self.u_std = train.groupby("user")["rating"].std()
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "log_n_u": np.log1p(rows["user"].map(self.n_u).fillna(0).to_numpy(dtype=float)),
            "u_std": rows["user"].map(self.u_std).fillna(0).to_numpy(dtype=float),
            "log_n_i": np.log1p(rows["item"].map(self.n_i).fillna(0).to_numpy(dtype=float)),
        })


class StackedRating(RatingModel):
    """
    base + regressor(features), trained on the base model's residuals.

    fit(): (1) split train into tr / val; fit base and sources on tr; train the regressor on
           val residuals. (2) refit base and sources on all of train for prediction.
    Pass unfitted prototypes: each fit works on deep copies, so they are never shared.
    """

    def __init__(
        self,
        base: RatingModel,
        sources: list[FeatureSource],
        regressor,
        val_frac: float = 0.25,
        seed: int = 0,
        scale: tuple[float, float] = (1, 5)
    ) -> None:
        self.base = base
        self.sources = sources
        self.regressor = regressor
        self.val_frac = val_frac
        self.seed = seed
        self.scale = scale
        # fitted state (trailing _ as in sklearn)
        self.base_: RatingModel | None = None

    def _fit_parts(self, train: pd.DataFrame) -> tuple[RatingModel, list[FeatureSource]]:
        base = copy.deepcopy(self.base).fit(train)
        
        return base, [copy.deepcopy(s).fit(train, base) for s in self.sources]

    @staticmethod
    def _features(base, sources, rows: pd.DataFrame) -> tuple[NDArray, pd.DataFrame]:
        b = base.predict(rows)
        F = pd.concat([pd.DataFrame({"base": b})] +
                      [s.features(rows) for s in sources], axis=1)

        return b, F

    def fit(self, train: pd.DataFrame) -> Self:
        tr, val = random_split(train, val_frac=self.val_frac, seed=self.seed)
        base, sources = self._fit_parts(tr)
        b, F = self._features(base, sources, val)
        self.reg_ = clone(self.regressor).fit(F, val["rating"].to_numpy() - b)
        self.base_, self.sources_ = self._fit_parts(train)

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray:
        if self.base_ is None:
            raise RuntimeError("StackedRating.predict called before fit")
        b, F = self._features(self.base_, self.sources_, rows)

        return np.clip(b + self.reg_.predict(F), *self.scale)
