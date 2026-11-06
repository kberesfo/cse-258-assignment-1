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
  - item-item similarity to the user's history: sum, max and mean over history items, for each
    similarity in TASK2_SIMILARITIES (binary cosine: |U_i ∩ U_j| / sqrt(|U_i| |U_j|))
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
Public: BPR 0.9066 -> LR combiner 0.9302 -> + seen-by-user rule 0.9407.
Rule evidence: notebooks/rules.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from assignment_1.modules.models import (
    BPRModel,
    ItemSimilarity,
    Similarity,
    apply_rules,
    binary_cosine,
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
# (similarity function, column prefix): each adds {prefix}_sum / _max / _mean
SimSpec = tuple[Similarity, str] | tuple[Similarity, str, tuple[int, ...]]
TASK2_SIMILARITIES: list[SimSpec] = [(binary_cosine, "cos")]
TASK2_TRAIN_FILE = "tasks_1_2_train.csv"
TASK2_TEST_FILE = "task_2_test.csv"


def task2_pair_features(
    train_df: pd.DataFrame,
    pairs: pd.DataFrame,
    similarities: list[SimSpec] = TASK2_SIMILARITIES
) -> pd.DataFrame:
    """Fit base models on train_df; return φ(u, item1) − φ(u, item2) per pair, one named column
    per feature: bpr, log_pop, then {prefix}_sum / _max / _mean for each similarity."""
    bpr = BPRModel(**TASK2_BPR_PARAMS).fit(train_df)
    item_tot = train_df["item"].value_counts()
    sims = [ItemSimilarity(f, prefix=p, top_k=k[0] if k else ()).fit(train_df)
            for f, p, *k in similarities]

    def cand(items: pd.Series) -> pd.DataFrame:
        base = pd.DataFrame({
            "bpr": bpr(pairs["user"].to_numpy(), items.to_numpy()),
            "log_pop": np.log1p(items.map(item_tot).fillna(0).to_numpy(dtype=float)),
        })
        return pd.concat([base] + [s.features(pairs["user"], items) for s in sims], axis=1)

    return cand(pairs["item1"]) - cand(pairs["item2"])


def run_task2(data_dir=None, out_dir="dist"):
    train = load_csv(TASK2_TRAIN_FILE, data_dir)
    test = load_csv(TASK2_TEST_FILE, data_dir)

    # Learn θ on validation pairs (base models fit on tr only)
    tr, pairs = make_pair_validation(train, seed=0)
    X_val = task2_pair_features(tr, pairs).to_numpy()
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = LogisticRegression(max_iter=1000).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = task2_pair_features(train, test).to_numpy()
    preds = np.where(combiner.predict(X_test) == 1,
                     test["item1"], test["item2"])

    # Last: test-design rules (see Rules in the docstring)
    preds = apply_rules(preds, unseen_item_rule(
        train, test), seen_by_user_rule(train, test))
    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
