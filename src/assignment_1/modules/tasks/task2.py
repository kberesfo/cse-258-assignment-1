"""
Task 2: read prediction — which of (item1, item2) did the user interact with? Metric: accuracy.

Test design: decoys popularity-matched, so popularity alone scores ~50%; no timestamps.

Model: pairwise logistic regression, P(item1) = σ(θ·[φ(u, item1) − φ(u, item2)]). Candidate
features φ: BPR score (implicit; item bias + latent factors), log popularity, and item-item
cosine similarity to the user's training history (sum, max, mean over history items;
cos(i, j) = |U_i ∩ U_j| / sqrt(|U_i||U_j|)). θ is fit on validation pairs built like the test,
with base models fit on the reduced train; test features use base models refit on all of
tasks_1_2_train.csv.

Validation (notebooks/task2.ipynb; held-out half of validation pairs): tuned BPR 0.9146,
BPR + popularity combiner 0.9146 (no gain: decoys are popularity-matched), + cosine 0.9371
(GB 0.9373 — no gain from trees, so the linear course model is used). Public (BPR only): 0.9066.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from assignment_1.modules.models import bpr_scorer, cos_features, fit_bpr, fit_item_cosine
from assignment_1.modules.util import load_csv, make_pair_validation, write_predictions

TASK2_BPR_PARAMS = {
    "factors": 100,
    "regularization": 0.003,
    "learning_rate": 0.01,
    "iterations": 100,
}
TASK2_FEATURES = ["bpr", "log_pop", "cos_sum", "cos_max", "cos_mean"]
TASK2_TRAIN_FILE = "tasks_1_2_train.csv"
TASK2_TEST_FILE = "task_2_test.csv"


def task2_pair_features(train_df: pd.DataFrame, pairs: pd.DataFrame) -> NDArray:
    """Fit base models on train_df; return φ(u, item1) − φ(u, item2) for each pair."""
    bpr = bpr_scorer(fit_bpr(train_df, **TASK2_BPR_PARAMS))
    item_tot = train_df["item"].value_counts()
    cos_fit = fit_item_cosine(train_df)

    def cand(items: pd.Series) -> pd.DataFrame:
        F = cos_features(cos_fit, pairs["user"], items)
        F["bpr"] = bpr(pairs["user"].to_numpy(), items.to_numpy())
        F["log_pop"] = np.log1p(
            items.map(item_tot).fillna(0).to_numpy(dtype=float))
        return F[TASK2_FEATURES]

    return (cand(pairs["item1"]) - cand(pairs["item2"])).to_numpy()


def run_task2(data_dir=None, out_dir="dist"):
    train = load_csv(TASK2_TRAIN_FILE, data_dir)
    test = load_csv(TASK2_TEST_FILE, data_dir)

    # Learn θ on validation pairs (base models fit on tr only)
    tr, pairs = make_pair_validation(train, seed=0)
    X_val = task2_pair_features(tr, pairs)
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = LogisticRegression(max_iter=1000).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = task2_pair_features(train, test)
    preds = np.where(combiner.predict(X_test) == 1,
                     test["item1"], test["item2"])
    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
