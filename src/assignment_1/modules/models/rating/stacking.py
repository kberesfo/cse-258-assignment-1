import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import clone

from assignment_1.modules.util import clip_timestamps, random_split

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


class RatingProfile(FeatureSource):
    """
    How each user and item rates, and when they were active:
        u_mean, i_mean, i_std        mean rating (and item spread)
        u_low, i_low                 share of 1–2 star ratings
        u_five, i_five               share of 5-star ratings
        u_first, u_last, i_first     first / last activity (years since 1970, clipped)
    The low-rating shares target the 1–2 star rows, where the base model errs most. NaN for
    unknown users or items (and i_std for single-rating items). Needs a timestamp column.
    """

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        by = {"u": train["user"], "i": train["item"]}
        r = train["rating"]
        self.tables = {}
        for name, key in by.items():
            self.tables[f"{name}_mean"] = (name, r.groupby(key).mean())
            self.tables[f"{name}_low"] = (name, (r <= 2).groupby(key).mean())
            self.tables[f"{name}_five"] = (name, (r == 5).groupby(key).mean())
        self.tables["i_std"] = ("i", r.groupby(by["i"]).std())
        years = clip_timestamps(train["timestamp"]) / (86400 * 365)
        self.tables["i_first"] = ("i", years.groupby(by["i"]).min())
        self.tables["u_first"] = ("u", years.groupby(by["u"]).min())
        self.tables["u_last"] = ("u", years.groupby(by["u"]).max())
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        key = {"u": rows["user"], "i": rows["item"]}
        return pd.DataFrame({name: key[side].map(tbl).to_numpy(dtype=float)
                             for name, (side, tbl) in self.tables.items()})


class StackedRating(RatingModel):
    """
    base + regressor(features), trained on the base model's residuals.

    fit(): (1) held-out residuals and features: base and sources are fit on one part of train
           and featurize the rest, so the regressor sees them as they look on unseen rows.
           n_folds=None: one tr / val split (val_frac); n_folds=k: k-fold out-of-fold, so the
           regressor trains on every row. Folds run in parallel (n_jobs).
           (2) base and sources refit on all of train for prediction.
    Predictions are clipped to `scale`. The regressor must accept NaN (HistGradientBoosting
    does). Pass unfitted prototypes: each fit works on deep copies, so they are never shared.
    """

    def __init__(
        self,
        base: RatingModel,
        sources: list[FeatureSource],
        regressor,
        val_frac: float = 0.25,
        seed: int = 0,
        scale: tuple[float, float] = (1, 5),
        n_folds: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.base = base
        self.sources = sources
        self.regressor = regressor
        self.val_frac = val_frac
        self.seed = seed
        self.scale = scale
        self.n_folds = n_folds
        self.n_jobs = n_jobs
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

    def _held_out(self, tr: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, NDArray]:
        """Features and residuals for rows, from base and sources fitted on tr only."""
        base, sources = self._fit_parts(tr)
        b, F = self._features(base, sources, rows)
        return F, rows["rating"].to_numpy() - b

    def _splits(self, train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        if self.n_folds is None:
            return [random_split(train, val_frac=self.val_frac, seed=self.seed)]
        fold = np.random.default_rng(self.seed).permutation(
            len(train)) % self.n_folds
        return [(train[fold != k].reset_index(drop=True), train[fold == k].reset_index(drop=True))
                for k in range(self.n_folds)]

    def fit(self, train: pd.DataFrame) -> Self:
        parts = Parallel(n_jobs=self.n_jobs)(
            delayed(self._held_out)(tr, rows) for tr, rows in self._splits(train))
        F = pd.concat([p[0] for p in parts], ignore_index=True)
        self.reg_ = clone(self.regressor).fit(
            F, np.concatenate([p[1] for p in parts]))
        self.base_, self.sources_ = self._fit_parts(train)

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray:
        if self.base_ is None:
            raise RuntimeError("StackedRating.predict called before fit")
        b, F = self._features(self.base_, self.sources_, rows)

        return np.clip(b + self.reg_.predict(F), *self.scale)


class ModelPrediction(FeatureSource):
    """Another rating model's prediction as a stacker feature (ensembling, as in lecture).

    The model is fit on the same training rows as the stack's base, and its prediction becomes
    one column. Pass an unfitted prototype; each fit works on a deep copy.
    """

    def __init__(self, model: RatingModel, name: str) -> None:
        self.model, self.name = model, name
        self.fitted: RatingModel | None = None

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        self.fitted = copy.deepcopy(self.model).fit(train)
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.fitted is None:
            raise RuntimeError("ModelPrediction.features called before fit")
        return pd.DataFrame({self.name: self.fitted.predict(rows)})
