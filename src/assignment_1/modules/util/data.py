"""Data loading, validation splits, and prediction-file writing (shared by all tasks)."""

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

ID_DTYPES = {"user": str, "item": str, "item1": str, "item2": str}

PathLike = str | Path

# before Goodreads launched (2007)
TS_MIN = pd.Timestamp("2005-01-01").timestamp()
# after the latest real timestamps (Dec 2017)
TS_MAX = pd.Timestamp("2018-12-31").timestamp()


def clip_timestamps(ts: pd.Series, lo: float = TS_MIN, hi: float = TS_MAX) -> pd.Series:
    """Clip corrupt timestamps (~0.14% of rows, e.g. year 25 / 3645) into the plausible range."""
    return ts.clip(lo, hi)


def data_path(name: str, data_dir: PathLike | None = None) -> Path:
    """Resolve a data file: explicit data_dir > $A1_DATA_DIR > ./data"""
    return Path(data_dir or os.environ.get("A1_DATA_DIR", "data")) / name


def load_csv(name: str, data_dir: PathLike | None = None) -> pd.DataFrame:
    """Load a train/test CSV with user/item IDs kept as strings (never parsed as numbers)."""
    path = data_path(name, data_dir)
    header = pd.read_csv(path, nrows=0).columns

    return pd.read_csv(path, dtype={c: t for c, t in ID_DTYPES.items() if c in header})


def random_split(
    df: pd.DataFrame, val_frac: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle rows and hold out val_frac as a validation set. Fixed seed = same split every run.

    Returns (train, val).
    """
    idx = np.random.default_rng(seed).permutation(len(df))
    n_val = int(len(df) * val_frac)
    train = df.iloc[idx[n_val:]].reset_index(drop=True)
    val = df.iloc[idx[:n_val]].reset_index(drop=True)
    return train, val


def holdout_last_per_user(
    df: pd.DataFrame, time_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out each user's most recent interaction (time-aware / sequential tasks).

    Returns (train, val).
    """
    df = df.sort_values(["user", time_col], kind="stable")
    is_last = ~df["user"].duplicated(keep="last")

    return df[~is_last].reset_index(drop=True), df[is_last].reset_index(drop=True)


def write_predictions(
    test_name: str,
    preds: Iterable[float | str],
    out_dir: PathLike = "dist",
    data_dir: PathLike | None = None,
) -> Path:
    """Write the Gradescope submission file for one task.

    test_name: e.g. "task_1_test.csv"; output is "task_1_predictions.csv" in out_dir.
    preds: one prediction per test row, in test-file order — a number (rating/delta tasks)
           or the chosen item ID (pairwise tasks).
    Format matches baselines.py: test header unchanged, then each test row + ",<pred>".
    """
    lines = data_path(test_name, data_dir).read_text().splitlines()
    header, rows = lines[0], [line for line in lines[1:] if line.strip()]
    preds = list(preds)
    if len(preds) != len(rows):
        raise ValueError(
            f"{test_name}: {len(rows)} test rows but {len(preds)} predictions"
        )

    out_path = Path(out_dir) / test_name.replace("_test", "_predictions")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header + "\n")
        f.writelines(
            f"{line.strip()},{p}\n" for line, p in zip(rows, preds, strict=True)
        )
    return out_path

def drop_zero_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rating-0 rows (Goodreads 'shelved, not rated'); Assignment 1 skips them in Tasks 1/3 MSE."""
    return df[df["rating"] > 0].reset_index(drop=True)