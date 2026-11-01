"""Bias-only model: f(u,i) = alpha + beta_u + beta_i."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from surprise import BaselineOnly

from .lfm import RatingScale, to_surprise_trainset


def fit_bias(
    train_df: pd.DataFrame,
    reg_u: float,
    reg_i: float,
    n_epochs: int = 25,
    rating_scale: RatingScale | None = None,
) -> BaselineOnly:
    """Fit alpha + beta_u + beta_i with separate user/item regularization (ALS)."""
    # (min,max)
    if rating_scale is None:
        rating_scale = (
            float(train_df["rating"].min()),
            float(train_df["rating"].max()),
        )
    # f(u,i) = \hat{r}_{ui} = b_{ui} = \mu + b_u + b_i =
    model = BaselineOnly(
        bsl_options={
            "method": "als",
            "reg_u": reg_u,
            "reg_i": reg_i,
            "n_epochs": n_epochs,
        },
        verbose=False,
    )

    model.fit(to_surprise_trainset(train_df, rating_scale))

    return model


def predict_bias(
    model: BaselineOnly, users: Iterable[str], items: Iterable[str]
) -> NDArray[np.float64]:
    """Predicted ratings for aligned users/items (unseen IDs -> their bias is 0)."""
    return np.array(
        [model.predict(u, i).est for u, i in zip(users, items, strict=True)]
    )
