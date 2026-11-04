"""
Task 5: next-item prediction — which of (item1, item2) is the user's next interaction after
their training history? Metric: accuracy. Test rows have no timestamp.

Test design: one row per user (86,786); decoys popularity-matched (medians ~100 vs 91
for interactions), so popularity alone scores ~50%.

Model: gradient-boosted pairwise combiner (same structure as Task 4). Each candidate gets
features from training history only; the query time is the user's last training interaction
(t_last), since the next item follows it. Features: BPR score, log popularity, item age and
not-yet-seen relative to t_last, trend share in the t_last time bin at 20/50/100 quantile bins,
and log first-order transition count from the user's last item. HistGradientBoostingClassifier
is trained on feature differences for validation pairs, with base models fit on the reduced
train; test features use base models refit on all of task_5_train.csv.

Rule (applied last): if exactly one candidate never appears in training, predict it. Decoys
are drawn from training interactions, so an unseen candidate must be the real next item.
Applies to ~3.96% of test rows. Validation drops these rows, so the combiner never learns
this case and the rule is required.

Validation: each user's most recent interaction held out (Holdout.LAST), paired with a
popularity-sampled decoy (notebooks/task5.ipynb). Baseline BPR (5 factors) 0.8425 -> tuned
BPR 0.8704 -> + trend/transition rules 0.8728 -> GB combiner 0.8932 (held-out half of
validation pairs). Public: rules 0.8703.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from assignment_1.modules.models import (
    apply_rules,
    fit_pair_context,
    pair_features,
    unseen_item_rule,
)
from assignment_1.modules.util import (
    Holdout,
    load_csv,
    make_pair_validation,
    write_predictions,
)

TASK5_BPR_PARAMS = {
    "factors": 50,
    "regularization": 0.005,
    "learning_rate": 0.01,
    "iterations": 100,
}
TASK5_TRAIN_FILE = "task_5_train.csv"
TASK5_TEST_FILE = "task_5_test.csv"


def run_task5(data_dir=None, out_dir="dist"):
    train = load_csv(TASK5_TRAIN_FILE, data_dir)
    test = load_csv(TASK5_TEST_FILE, data_dir)

    # Learn the combiner on validation pairs (base models fit on tr only; no timestamp -> t_last)
    tr, pairs = make_pair_validation(train, holdout=Holdout.LAST, seed=0)
    X_val = pair_features(fit_pair_context(tr, TASK5_BPR_PARAMS, transitions=True), pairs)
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = HistGradientBoostingClassifier(random_state=0).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = pair_features(fit_pair_context(train, TASK5_BPR_PARAMS, transitions=True), test)
    preds = np.where(combiner.predict(X_test) == 1, test["item1"], test["item2"])

    # Last: an unseen candidate is the real item (case absent from validation)
    preds = apply_rules(preds, unseen_item_rule(train, test))
    return write_predictions(TASK5_TEST_FILE, preds, out_dir, data_dir)