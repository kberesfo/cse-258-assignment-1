"""
Task 4: interaction prediction with timestamps — which of (item1, item2) did the user
interact with at time t? Metric: accuracy.

Model: BPR (implicit; item bias + user/item latent factors) trained on ALL of
tasks_3_4_train.csv, plus a time rule: if exactly one candidate has no training interactions
before t (first appearance after t), predict the other one. Otherwise use the higher BPR score
(ties -> item1). Timestamps are clipped to 2005-2018 (~0.14% are corrupt, e.g. year 25 / 3645).

Test design: decoys appear to be sampled ignoring time, so in ~25% of test rows exactly one
item did not exist yet at t; on validation that rule is right 97% of the time where it applies.

Validation: held-out 10% of interactions with popularity-sampled decoys and timestamps
(notebooks/task4.ipynb; first-appearance times computed from the training part only).
Baseline config (5 factors) 0.8161 -> tuned BPR (50 factors, reg 0.01) 0.8613
-> BPR + time rule 0.8812 (+8.0%).
"""

from assignment_1.modules.models import (
    apply_rules,
    bpr_scorer,
    fit_bpr,
    not_yet_seen_rule,
)
from assignment_1.modules.util import load_csv, predict_pairs, write_predictions

TASK4_BPR_PARAMS = {
    "factors": 50,
    "regularization": 0.010,
    "learning_rate": 0.01,
    "iterations": 100,
}
TASK4_TRAIN_FILE = "tasks_3_4_train.csv"
TASK4_TEST_FILE = "task_4_test.csv"


def run_task4(data_dir=None, out_dir="dist"):
    train = load_csv(TASK4_TRAIN_FILE, data_dir)
    test = load_csv(TASK4_TEST_FILE, data_dir)

    preds = predict_pairs(test, bpr_scorer(
        fit_bpr(train, **TASK4_BPR_PARAMS)))   # BPR's choice per row
    preds = apply_rules(preds, not_yet_seen_rule(
        train, test))                    # time rule overrides

    return write_predictions(TASK4_TEST_FILE, preds, out_dir, data_dir)
