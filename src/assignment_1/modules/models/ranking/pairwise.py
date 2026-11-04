from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from assignment_1.modules.util import clip_timestamps, time_bin_edges, time_bins

from .bpr import bpr_scorer, fit_bpr
from .markov import fit_transitions, transition_scorer


@dataclass
class PairContext:
    """Base models and statistics for candidate features, all fitted on one training set."""

    bpr: Callable
    # item -> first interaction time
    first_seen: pd.Series
    # item -> interaction count
    item_tot: pd.Series
    # n_bins -> (edges, (item, bin) counts)
    trend_tables: dict[int, tuple[NDArray, pd.Series]]
    # user -> last interaction time (query time when pairs have no timestamp)
    t_last: pd.Series
    # fallback query time for users with no history
    t_max: float
    # (user's last item -> candidate) transition counts; Task 5 only
    trans: Callable | None = None


def fit_pair_context(
    train_df: pd.DataFrame,
    bpr_params: dict,
    trend_bins=(20, 50, 100),
    transitions: bool = False,
) -> PairContext:
    ts = clip_timestamps(train_df["timestamp"])

    trend_tables = {}
    for nb in trend_bins:
        edges = time_bin_edges(train_df["timestamp"], nb)
        bins = time_bins(train_df["timestamp"], edges)
        trend_tables[nb] = (edges, train_df.groupby([train_df["item"], bins]).size())

    return PairContext(
        bpr=bpr_scorer(fit_bpr(train_df, **bpr_params)),
        first_seen=ts.groupby(train_df["item"]).min(),
        item_tot=train_df["item"].value_counts(),
        trend_tables=trend_tables,
        t_last=train_df.groupby("user")["timestamp"].max(),
        t_max=float(train_df["timestamp"].max()),
        trans=transition_scorer(fit_transitions(train_df)) if transitions else None,
    )


def cand_features(ctx: PairContext, users: pd.Series, items: pd.Series, ts: pd.Series) -> pd.DataFrame:
    t = clip_timestamps(ts).to_numpy(dtype=float)
    fs = items.map(ctx.first_seen).to_numpy(dtype=float)
    tot = items.map(ctx.item_tot).fillna(0).to_numpy(dtype=float)
    age = t - fs

    feats = {
        "bpr": ctx.bpr(users.to_numpy(), items.to_numpy()),
        "not_yet": (age < 0).astype(float),
        "log_age": np.log1p(np.clip(np.nan_to_num(age, nan=0.0), 0, None) / 86400),
        "log_pop": np.log1p(tot),
    }
    for nb, (edges, tbl) in ctx.trend_tables.items():
        idx = pd.MultiIndex.from_arrays([items, time_bins(ts, edges)])
        in_bin = tbl.reindex(idx, fill_value=0).to_numpy(dtype=float)
        feats[f"trend_{nb}"] = in_bin / (tot + 1)

    if ctx.trans is not None:
        feats["trans"] = np.log1p(ctx.trans(users.to_numpy(), items.to_numpy()))
        feats["unseen"] = (tot == 0).astype(float)
    return pd.DataFrame(feats)


def query_times(ctx: PairContext, pairs: pd.DataFrame) -> pd.Series:
    """Row timestamp when pairs have one (Task 4); else the user's last training time (Task 5)."""
    if "timestamp" in pairs:
        return pairs["timestamp"]
    return pairs["user"].map(ctx.t_last).fillna(ctx.t_max)


def pair_features(ctx: PairContext, pairs: pd.DataFrame) -> NDArray:
    """features(item1) - features(item2); positive values favour item1."""
    users, ts = pairs["user"], query_times(ctx, pairs)
    F1 = cand_features(ctx, users, pairs["item1"], ts)
    F2 = cand_features(ctx, users, pairs["item2"], ts)
    return (F1 - F2).to_numpy()