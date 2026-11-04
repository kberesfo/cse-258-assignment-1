"""
Task 4 ranks two candidates using a gradient-boosted pairwise combiner. 
Each candidate gets features computed from training history only: BPR score,
whether the item had not yet appeared at the query time, item age,
log popularity, and the item’s share of interactions in the query’s
time bin at three resolutions (20/50/100 quantile bins). The combiner
(HistGradientBoostingClassifier) is trained on feature differences for 
validation pairs built like the test (popularity-matched decoys), 
with base models fit on the reduced train. Test features come from 
base models refit on the full train. Held-out validation accuracy was 0.9041,
against 0.8803 for BPR plus the not-yet-seen rule.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from assignment_1.modules.models import (
    fit_pair_context,
    pair_features,
)
from assignment_1.modules.util import (
    load_csv,
    make_pair_validation,
    write_predictions,
)

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

    # Learn the combiner on validation pairs (base models fit on tr only)
    tr, pairs = make_pair_validation(train, include_timestamp=True, seed=0)
    X_val = pair_features(fit_pair_context(tr, TASK4_BPR_PARAMS), pairs)
    y_val = (pairs["which"] == pairs["item1"]).to_numpy().astype(int)
    combiner = HistGradientBoostingClassifier(random_state=0).fit(X_val, y_val)

    # Test features from base models refit on all of train
    X_test = pair_features(fit_pair_context(train, TASK4_BPR_PARAMS), test)
    preds = np.where(combiner.predict(X_test) == 1,
                     test["item1"], test["item2"])
    
    return write_predictions(TASK4_TEST_FILE, preds, out_dir, data_dir)
