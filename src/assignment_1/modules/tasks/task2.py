"""
Task 2: read prediction. Which of (item1, item2) did the user interact with?
Metric: accuracy (higher is better). Test rows have no timestamp; training timestamps are used
for time features relative to each user's last interaction.

Test design
-----------
Decoys are popularity-matched, so popularity alone scores ~50%. Every test candidate appears
in training. The decoy sampler does not exclude the user's own history (see Rules).

Model
-----
Gradient-boosted pairwise combiner on feature differences φ(u, item1) − φ(u, item2) (it was the
course's pairwise logistic regression σ(θ · Δφ) until EASE was added; see Validation).
Candidate features φ, computed from training history only:
  - BPR score, K = 100 (implicit; item bias + latent factors)
  - log popularity
  - item-item cosine similarity to the user's history: sum, max and mean over history items,
    cos(i, j) = |U_i ∩ U_j| / sqrt(|U_i| |U_j|)
  - time features relative to the user's last training interaction (as in Task 5): item age,
    not-yet-seen, trend share at 20/50/100 quantile bins
  - a second BPR with K = 10 (small K; an extra ensemble member, as suggested in lecture)
  - EASE (Steck 2019, λ=25): closed-form item-item autoencoder weights, top 200 per item;
    sum / max / mean over the user's history. 51k items, so fit in float32 (~10 GB, minutes).
  - time-local cosine (LocalSimilarity, from Task 4): rows have no timestamp, so the query time
    is the candidate's median read time; similarity to the user's 20 reads nearest it
    (if the user read the book, they likely read it when everyone else did).
The combiner is fit on validation pairs built like the test, with base models fit on the
reduced train. Test features come from base models refit on all of tasks_1_2_train.csv.

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
  -> + cosine 0.9371 (GB 0.9373: on these features trees added nothing, so LR was kept)
  -> + time features and BPR K=10: 0.9383 (+0.0006 to +0.0013 on four different held-out
     halves: small but consistent)
  -> + EASE + time-local cosine, GB combiner: +0.0027 mean over four halves (0.9378 -> 0.9404;
     notebooks/task2_t4ideas.ipynb). With LR the same features gave +0.0015: EASE gives the
     trees interactions to use. EASE λ 10/25/100 equal under GB; time-local EASE +0.0001.
Diagnostics (notebooks/task2_diagnostics.ipynb): accuracy is flat across history length and
popularity gap; >80% of errors are low-confidence pairs where every feature alone is at chance.
Public: BPR 0.9066 -> LR combiner 0.9302 -> + seen-by-user rule 0.9407 -> + time, BPR K=10
0.9410 -> 0.9454 (EASE + time-local cosine, GB; +0.0044, more than validation's +0.0027).
Rule evidence: notebooks/rules.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from assignment_1.modules.models import (
    BPRModel,
    PairFeatures,
    apply_rules,
    binary_cosine,
    seen_by_user_rule,
    unseen_item_rule,
)
from assignment_1.modules.util import load_csv, make_pair_validation, write_predictions

TASK2_BPR_PARAMS = {
    "factors": 100, "regularization": 0.003, "learning_rate": 0.01, "iterations": 100
}
TASK2_BPR_SMALL_PARAMS = {
    "factors": 10, "regularization": 0.01, "learning_rate": 0.01, "iterations": 100
}
TASK2_EASE_LAM = 25.0
TASK2_TRAIN_FILE = "tasks_1_2_train.csv"
TASK2_TEST_FILE = "task_2_test.csv"


def task2_pair_features(train_df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """φ(u, item1) − φ(u, item2): PairFeatures (BPR K=100, popularity, item age / not-yet-seen /
    trends relative to the user's last interaction, cosine similarity to history, time-local
    cosine at the candidate's median read time, EASE) plus a second BPR with K=10."""
    X = PairFeatures(TASK2_BPR_PARAMS, similarity=binary_cosine, sim_prefix="cos",
                     local_similarity=True, local_time="item_median",
                     ease_lam=TASK2_EASE_LAM, ease_dtype=np.float32,
                     ).fit(train_df).transform(pairs)
    small = BPRModel(**TASK2_BPR_SMALL_PARAMS).fit(train_df)
    u = pairs["user"].to_numpy()
    d_small = small(u, pairs["item1"].to_numpy()) - \
        small(u, pairs["item2"].to_numpy())
    
    return np.column_stack([X, d_small])


def run_task2(data_dir=None, out_dir="dist"):
    train = load_csv(TASK2_TRAIN_FILE, data_dir)
    test = load_csv(TASK2_TEST_FILE, data_dir)

    # Learn the combiner on validation pairs (base models fit on tr only)
    tr, pairs = make_pair_validation(train, seed=0)
    X_val = task2_pair_features(tr, pairs)
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = HistGradientBoostingClassifier(random_state=0).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = task2_pair_features(train, test)
    preds = np.where(combiner.predict(X_test) == 1,
                     test["item1"], test["item2"])

    # Last: test-design rules (see Rules in the docstring)
    preds = apply_rules(preds, unseen_item_rule(
        train, test), seen_by_user_rule(train, test))
    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
