"""
Task 6: time-delta prediction — how long (seconds) until the user's next interaction? Metric: MAE.

Model: per-user median of each user's last 10 gaps between consecutive interactions, shrunk
toward the global median gap in log space: log(pred) = w*log(user median) + (1-w)*log(global
median), w = n_gaps / (n_gaps + 1.5). Trained on ALL of task_6_train.csv.

Metric inferred as MAE: on validation the baseline's MAE (12.05M) matches the public baseline
(13.56M) in scale, while its MSE (8e14) is eight orders of magnitude off. Gaps are heavily
skewed (median 13 days, mean 71 days), so medians, not means, and shrinkage for users with few
gaps account for most of the gain.

Validation: each user's last gap held out, predicted from their earlier gaps
(notebooks/task6.ipynb). Baseline per-user median 12,052,136 -> shrunk median of last 10 gaps
(lam 1.5) 10,137,22x (+15.9%).
"""

from assignment_1.modules.models import (
    fit_gap_model,
    predict_gaps,
)
from assignment_1.modules.util import load_csv, write_predictions

TASK6_GAP_PARAMS = {"k": 10, "lam": 1.5}

TASK6_TRAIN_FILE = "task_6_train.csv"
TASK6_TEST_FILE = "task_6_test.csv"


def run_task6(data_dir=None, out_dir="dist"):
    train = load_csv(TASK6_TRAIN_FILE, data_dir)
    test = load_csv(TASK6_TEST_FILE, data_dir)

    preds = predict_gaps(
        fit_gap_model(
            train, **TASK6_GAP_PARAMS
        ),
        test["user"]
    )
    return write_predictions(TASK6_TEST_FILE, preds, out_dir, data_dir)
