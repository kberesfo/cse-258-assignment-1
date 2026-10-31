"""Latent factor model via Surprise's SVD.

Despite the name this is NOT a true SVD: it's the Funk-style model
    f(u,i) = alpha + beta_u + beta_i + gamma_u . gamma_i
fit by SGD on observed ratings only (PML §5.1, eqs. 2.4-2.5 in the notes).
"""

import numpy as np
from surprise import SVD, Dataset, Reader


def to_surprise_trainset(df, rating_scale):
    """DataFrame with user/item/rating columns -> Surprise trainset using every row."""
    reader = Reader(rating_scale=rating_scale)
    return Dataset.load_from_df(
        df[["user", "item", "rating"]], reader
    ).build_full_trainset()


def fit_svd(train_df, rating_scale=None, seed=0, **svd_params):
    """Fit Surprise SVD on all rows of train_df.

    rating_scale: (min, max) predictions get clipped to. Defaults to the range seen in train_df;
    pass it explicitly when train_df is a validation-split subset so it matches the full data.
    svd_params: n_factors, n_epochs, lr_all, reg_all, ... (Surprise SVD kwargs).
    """
    if rating_scale is None:
        rating_scale = (train_df["rating"].min(), train_df["rating"].max())
    model = SVD(random_state=seed, **svd_params)
    model.fit(to_surprise_trainset(train_df, rating_scale))
    return model


def predict_svd(model, users, items):
    """Predicted ratings for aligned sequences of users and items (unseen IDs -> global mean)."""
    return np.array([model.predict(u, i).est for u, i in zip(users, items)])
