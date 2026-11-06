"""
Task 4: time-aware read prediction. At the row's timestamp, which of (item1, item2) did the
user interact with? Metric: accuracy (higher is better).

Test design
-----------
Decoys are popularity-matched, so popularity alone scores ~50%. Decoys ignore time: a decoy can
be a book that did not yet exist at the row's timestamp, while the real item always did
(~26% of rows have exactly one not-yet-seen candidate). Every test candidate appears in
training. The decoy sampler does not exclude the user's own history (see Rules).

Model
-----
Gradient-boosted pairwise combiner on PairFeatures. Each candidate gets features computed from
training history only, at the row's timestamp t:
  - BPR score (implicit; item bias + latent factors)
  - not-yet-seen: the item's first training interaction is after t
  - item age at t (log days since first interaction)
  - log popularity
  - trend: the item's share of interactions in t's time bin, at 20/50/100 quantile bins
HistGradientBoostingClassifier is trained on feature differences (item1 - item2) for
validation pairs, with base models fit on the reduced train. Test features come from base
models refit on all of tasks_3_4_train.csv. The earlier hard not-yet-seen rule is now a
feature, so the combiner weighs it against the other signals.

Why gradient boosting: the pairwise logistic regression fits one weight per feature,
so each signal contributes linearly and independently. Boosted trees fit the same log-loss
objective but replace θ·Δφ with a sum of small trees, which can learn interactions such as
"the BPR gap matters less when one candidate is not-yet-seen". On the held-out half this gave
0.9041 vs 0.9018 for LR.

Rules
-----
Applied last. Neither case occurs in validation, so the combiner cannot learn them.
  1. Unseen item: if exactly one candidate never appears in training, predict it. Kept as a
     guard; fires on 0% of this test (every candidate appears in training).
  2. Seen by user: if exactly one candidate is already in the user's training history, predict
     the other. Positives are held out of the history, and the test's decoy sampler does not
     exclude it, so a history item is the decoy. 5.07% of test rows.

Validation
----------
make_pair_validation(train, include_timestamp=True, seed=0): random held-out interactions
with their timestamps, paired with popularity-sampled decoys that exclude the user's history
(notebooks/task4.ipynb). Accuracy on the held-out half of validation pairs:
  BPR + not-yet-seen rule 0.8803 -> LR combiner 0.9018 -> GB combiner 0.9041
(trend at three resolutions beat a single 50-bin trend: GB 0.9026 -> 0.9041).
Public: BPR + rule 0.8706 -> GB combiner 0.8984 -> + seen-by-user rule 0.9172
(predicted +1.97 from rules.ipynb; observed +1.88).
Rule evidence: notebooks/rules.ipynb.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from assignment_1.modules.models import (
    PairFeatures,
    apply_rules,
    binary_cosine,
    seen_by_user_rule,
    unseen_item_rule,
)
from assignment_1.modules.util import load_csv, make_pair_validation, write_predictions

TASK4_BPR_PARAMS = {
    "factors": 50,
    "regularization": 0.010,
    "learning_rate": 0.01,
    "iterations": 100,
}
TASK4_TRAIN_FILE = "tasks_3_4_train.csv"
TASK4_TEST_FILE = "task_4_test.csv"


def task4_features() -> PairFeatures:
    """Unfitted feature builder; fit on tr for validation, on all of train for the test."""
    return PairFeatures(TASK4_BPR_PARAMS, similarity=binary_cosine, sim_prefix="cos")


def run_task4(data_dir=None, out_dir="dist"):
    train = load_csv(TASK4_TRAIN_FILE, data_dir)
    test = load_csv(TASK4_TEST_FILE, data_dir)

    # Learn the combiner on validation pairs (base models fit on tr only)
    tr, pairs = make_pair_validation(train, include_timestamp=True, seed=0)
    X_val = task4_features().fit(tr).transform(pairs)
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = HistGradientBoostingClassifier(random_state=0).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = task4_features().fit(train).transform(test)
    preds = np.where(combiner.predict(X_test) == 1,
                     test["item1"], test["item2"])

    # Last: test-design rules (see Rules in the docstring)
    preds = apply_rules(preds, unseen_item_rule(
        train, test), seen_by_user_rule(train, test))
    return write_predictions(TASK4_TEST_FILE, preds, out_dir, data_dir)
