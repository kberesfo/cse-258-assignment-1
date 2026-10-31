"""Data loading, validation splits, and prediction-file writing (shared by all tasks)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID_DTYPES = {"user": str, "item": str, "item1": str, "item2": str}


def data_path(name, data_dir=None):
    """Resolve a data file: explicit data_dir > $A1_DATA_DIR > ./data"""
    return Path(data_dir or os.environ.get("A1_DATA_DIR", "data")) / name


def load_csv(name, data_dir=None):
    """Load a train/test CSV with user/item IDs kept as strings (never parsed as numbers)."""
    path = data_path(name, data_dir)
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: t for c, t in ID_DTYPES.items() if c in header})


def random_split(df, val_frac=0.25, seed=0):
    """Shuffle rows and hold out val_frac as a validation set. Fixed seed = same split every run."""
    idx = np.random.default_rng(seed).permutation(len(df))
    n_val = int(len(df) * val_frac)
    return df.iloc[idx[n_val:]].reset_index(drop=True), df.iloc[
        idx[:n_val]
    ].reset_index(drop=True)


def holdout_last_per_user(df, time_col="timestamp"):
    """Hold out each user's most recent interaction (for the time-aware / sequential tasks)."""
    df = df.sort_values(["user", time_col], kind="stable")
    is_last = ~df["user"].duplicated(keep="last")
    return df[~is_last].reset_index(drop=True), df[is_last].reset_index(drop=True)


def write_predictions(test_name, preds, out_dir="dist", data_dir=None):
    """Write predictions in the baseline format: test header unchanged, each test row + ',<pred>'."""
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
        f.writelines(f"{line.strip()},{p}\n" for line, p in zip(rows, preds))

    return out_path
