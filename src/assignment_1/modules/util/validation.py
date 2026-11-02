from enum import StrEnum

import numpy as np
import pandas as pd

from .data import holdout_last_per_user, random_split


class Holdout(StrEnum):
    RANDOM = "random"  # random interactions (Tasks 2, 4)
    LAST = "last"  # each user's most recent interaction (Task 5)


def make_pair_validation(
    train: pd.DataFrame,
    holdout: Holdout = Holdout.RANDOM,
    val_frac: float = 0.1,
    seed: int = 0,
    include_timestamp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a validation set shaped like a pairwise test file.

    Holds out interactions (random rows, or each user's last one), drops held-out rows whose
    user/item is absent from the training part, and pairs each with a popularity-sampled decoy
    the user never interacted with (checked against the full train). Positions are randomized.

    Returns (tr, pairs): tr = training part — fit models on tr ONLY;
    pairs = user, item1, item2, which (the real item).
    val_frac applies only to Holdout.RANDOM; Holdout.LAST holds out one item per user.
    """
    holdout = Holdout(
        holdout
    )  # also accepts "random"/"last"; raises ValueError otherwise

    match holdout:
        case Holdout.RANDOM:
            tr, val = random_split(train, val_frac=val_frac, seed=seed)
        case Holdout.LAST:
            tr, val = holdout_last_per_user(train)

    seen = val.user.isin(tr.user) & val.item.isin(tr.item)
    val = val[seen].reset_index(drop=True)

    history = (
        train.groupby("user")["item"].apply(set).to_dict()
    )  # dict[user] = set[item]
    users = val.user.to_numpy()  # pair k -> user ID
    train_items = train.item.to_numpy()  # train row -> item ID
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(train), size=len(val))
    decoys = train.item.to_numpy()[idx]

    bad = np.array([i for i, d in enumerate(decoys) if d in history[users[i]]]).astype(
        dtype=int
    )

    while bad.size > 0:
        # pull a new random item for each bad pair
        for i in bad:
            decoys[i] = train_items[rng.integers(len(train))]

        # keep only the ones that collided again
        bad = np.array([i for i in bad if decoys[i] in history[users[i]]], dtype=int)

    assert not any(d in history[u] for u, d in zip(users, decoys))
    assert (val.item.to_numpy() != decoys).all()

    real = val.item.to_numpy()
    real_first = rng.random(len(val)) < 0.5  # coin flip per pair

    pairs = pd.DataFrame(
        index=range(len(val)), columns=["user", "item1", "item2", "which"]
    )
    pairs["user"] = users
    pairs["which"] = real
    pairs["item1"] = np.where(real_first, real, decoys)
    pairs["item2"] = np.where(real_first, decoys, real)

    if include_timestamp:
        pairs["timestamp"] = val["timestamp"].to_numpy()

    assert (pairs.item1 == pairs.item2).sum() == 0  # should be 0
    frac = (pairs.which == pairs.item1).mean()
    assert abs(frac - 0.5) < 0.01, (
        f"real item in item1 {frac:.3f} of the time"
    )  # should be ≈ 0.5

    return tr, pairs
