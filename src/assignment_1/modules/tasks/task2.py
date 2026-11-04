"""
Task 2: read prediction. Which of (item1, item2) did the user interact with?
Metric: accuracy (higher is better). No timestamps are used.

Test design
-----------
Decoys are popularity-matched, so popularity alone scores ~50%. Every test candidate appears
in training. The decoy sampler does not exclude the user's own history (see Rules).

Model
-----
Pairwise logistic regression (the course's pairwise model):
    P(item1) = σ(θ · [φ(u, item1) − φ(u, item2)])
Candidate features φ, computed from training history only:
  - BPR score (implicit; item bias + latent factors)
  - log popularity
  - item-item cosine similarity to the user's history: sum, max and mean over history items,
    with cos(i, j) = |U_i ∩ U_j| / sqrt(|U_i| |U_j|)
θ is fit on validation pairs built like the test, with base models fit on the reduced train.
Test features come from base models refit on all of tasks_1_2_train.csv.

Rules
-----
Applied last. Neither case occurs in validation, so the model cannot learn them.
  1. Unseen item: if exactly one candidate never appears in training, predict it. Kept as a
     guard; fires on 0% of this test (every candidate appears in training).
  2. Seen by user: if exactly one candidate is already in the user's training history, predict
     the other. Positives are held out of the history, and the test's decoy sampler does not
     exclude it, so a history item is the decoy. 1.92% of test rows.

Validation
----------
make_pair_validation(train, seed=0): held-out interactions paired with popularity-sampled
decoys that exclude the user's history (notebooks/task2.ipynb). Accuracy on the held-out half
of validation pairs:
  tuned BPR 0.9146 -> BPR + popularity combiner 0.9146 (no gain: decoys are popularity-matched)
  -> + cosine 0.9371 (GB 0.9373; trees add nothing, so the linear course model is used)
Diagnostics (notebooks/task2_diagnostics.ipynb): accuracy is flat across history length and
popularity gap; >80% of errors are low-confidence pairs where every feature alone is at chance.
Public: BPR 0.9066 -> LR combiner 0.9302 -> + seen-by-user rule (expected ~0.940).
Rule evidence: notebooks/rules.ipynb.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from assignment_1.modules.models import (
    apply_rules,
    bpr_scorer,
    cos_features,
    fit_bpr,
    fit_item_cosine,
    seen_by_user_rule,
    unseen_item_rule,
)
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

    preds = apply_rules(
        preds,
        unseen_item_rule(train, test),
        seen_by_user_rule(train, test)
    )
    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
