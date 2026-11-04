import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from assignment_1.modules.util import clip_timestamps, time_bin_edges, time_bins

from .markov import TransitionFit, transition_scorer


def not_yet_seen_rule(
    train_df: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[NDArray[np.bool_], NDArray]:
    """
    Where exactly one candidate's first interaction in train_df is after the row's timestamp,
    choose the other one (Task 4: decoys ignore time). Needs a timestamp column in pairs.

    Returns (applies, choice): rows the rule covers, and the item it picks there.
    """
    first_seen = clip_timestamps(
        train_df["timestamp"]
    ).groupby(train_df["item"]).min()

    t = clip_timestamps(pairs["timestamp"])

    early1 = pairs["item1"].map(first_seen) > t
    early2 = pairs["item2"].map(first_seen) > t

    applies = (early1 ^ early2).to_numpy()
    choice = np.where(early1, pairs["item2"], pairs["item1"])

    return applies, choice


def unseen_item_rule(train_df: pd.DataFrame, pairs: pd.DataFrame) -> tuple[NDArray[np.bool_], NDArray]:
    """Task 5: where exactly one candidate never appears in train_df, choose it
    (decoys are drawn from training interactions, so an unseen candidate is the real item)."""

    unseen1 = ~pairs["item1"].isin(train_df["item"])
    unseen2 = ~pairs["item2"].isin(train_df["item"])
    applies = (unseen1 ^ unseen2).to_numpy()
    # item1 is the unseen one -> pick it
    choice = np.where(unseen1, pairs["item1"], pairs["item2"])

    return applies, choice


def seen_by_user_rule(train_df: pd.DataFrame, pairs: pd.DataFrame) -> tuple[NDArray[np.bool_], NDArray]:
    """
    If exactly one candidate is already in this user's training history, predict the other.
    Test positives are held-out interactions, so they never appear in the user's own history;
    a candidate that does is the decoy (the test's decoy sampler did not exclude history).
    """
    hist = pd.MultiIndex.from_frame(train_df[["user", "item"]])
    seen1 = pd.MultiIndex.from_arrays([pairs["user"], pairs["item1"]]).isin(hist)
    seen2 = pd.MultiIndex.from_arrays([pairs["user"], pairs["item2"]]).isin(hist)
    # seen1 XOR seen2
    applies = seen1 ^ seen2
    choice = np.where(seen1, pairs["item2"], pairs["item1"])

    return applies, choice


def transition_rule(
    fit: TransitionFit, pairs: pd.DataFrame, min_gap: int = 5
) -> tuple[NDArray[np.bool_], NDArray]:
    """Where one candidate follows the user's last item at least `min_gap` more times than the
    other, choose it. fit = fit_transitions(training data) — tr for validation, full train for test.

    Returns (applies, choice): rows the rule covers, and the item it picks there.
    """
    tscore = transition_scorer(fit)
    users = pairs["user"].to_numpy()
    t1 = tscore(users, pairs["item1"].to_numpy())
    t2 = tscore(users, pairs["item2"].to_numpy())
    applies = np.abs(t1 - t2) >= min_gap
    choice = np.where(t1 >= t2, pairs["item1"], pairs["item2"])

    return applies, choice


def apply_rules(preds: ArrayLike, *rules: tuple[NDArray[np.bool_], NDArray]) -> NDArray:
    """Override model predictions with each rule's choice where it applies (later rules win)."""
    preds = np.asarray(preds)
    for applies, choice in rules:
        # rule's pick where it applies, else keep preds
        preds = np.where(applies, choice, preds)

    return preds


def trend_rule(
    train_df: pd.DataFrame, pairs: pd.DataFrame, min_gap: float = 0.25, n_bins: int = 50
) -> tuple[NDArray[np.bool_], NDArray]:
    """..."""
    # 1. time periods, from training timestamps
    edges = time_bin_edges(train_df["timestamp"], n_bins=n_bins)
    tr_bins = time_bins(train_df["timestamp"], edges)

    # 2. how many interactions each item has in each period, and in total
    item_bin = pd.Series(1, index=pd.MultiIndex.from_arrays(
        [train_df["item"], tr_bins])).groupby(level=[0, 1]).size()
    item_tot = train_df["item"].value_counts()

    # 3. each user's "current" period = the period of their last training interaction
    t_last = train_df.groupby("user")["timestamp"].max()
    ubin = time_bins(pairs["user"].map(t_last), edges)

    # 4. trend score per candidate = share of its interactions in the user's current period
    def trend(items):
        keys = pd.MultiIndex.from_arrays([items, ubin])
        in_bin = item_bin.reindex(keys, fill_value=0).to_numpy(dtype=float)
        return in_bin / (items.map(item_tot).fillna(0).to_numpy(dtype=float) + 1)

    f1, f2 = trend(pairs["item1"]), trend(pairs["item2"])

    # the rule: act only where the gap is big; pick the candidate with the higher trend score
    applies = np.abs(f1 - f2) >= min_gap
    choice = np.where(f1 >= f2, pairs["item1"], pairs["item2"])
    return applies, choice
