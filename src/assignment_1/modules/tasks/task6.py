"""
Task 6: time-delta prediction. How long (seconds) until the user's next interaction?
Metric: MAE (lower is better).

Test design
-----------
One row per user, with the item of the next interaction. Metric confirmed as MAE (on validation
the baseline's MAE matched the public baseline in scale; its MSE was orders of magnitude off).
Gaps are heavily skewed (median ~13 days, mean ~71 days), so medians, not means: MAE is
minimized by the median.

Model
-----
GapRegressor (time-series regression on the previous gaps, as suggested in lecture):
HistGradientBoostingRegressor with MAE loss on log1p(gap). Since median(log g) = log(median g),
fitting MAE in log space still estimates the median gap. Trained by stepping back one
interaction per user (each user's last training gap is the target, earlier history gives the
features). Features:
  - the last 5 log gaps; log-gap median / mean / std / count; median of the last 10
  - GapModel's prediction: per-user median of the last 10 gaps, shrunk toward the global median
    in log space, log ĝ = w log m_u + (1 − w) log m, w = n_u / (n_u + 1.5)
  - the item first-seen bound (days) and item popularity
Lower bound, applied to the output: the next book can't be read before it first appears in
training, so ĝ ≥ first_seen(item) − t_last(user) + 35 days (the median delay after a book first
appears; applies to ~13% of users). An upper bound from the end of the data hurt (corrupt future
timestamps), not used.

Validation
----------
Each user's last training gap held out, predicted from the earlier history
(notebooks/task6.ipynb):
  per-user median 12,052,136 -> shrunk median, last 10 gaps (λ 1.5) 10,137,223 (−15.9%)
  -> + first-seen bound 8,531,601 -> + 35-day offset, λ 2: ~8.43M
  -> GapRegressor (base λ 1.5) 8,012,169 (−34% vs the baseline)
Public: 13,013,427 -> 11,209,411 (bound + offset) -> <after upload> (GapRegressor).
"""

from assignment_1.modules.models import SECONDS_PER_DAY, GapModel, GapRegressor
from assignment_1.modules.util import load_csv, write_predictions

TASK6_GAP_PARAMS = {"k": 10, "lam": 1.5}

TASK6_TRAIN_FILE = "task_6_train.csv"
TASK6_TEST_FILE = "task_6_test.csv"
TASK6_BOUND_OFFSET = 35 * SECONDS_PER_DAY


def run_task6(data_dir=None, out_dir="dist"):
    train = load_csv(TASK6_TRAIN_FILE, data_dir)
    test = load_csv(TASK6_TEST_FILE, data_dir)

    model = GapRegressor(
        base=GapModel(
            **TASK6_GAP_PARAMS
        ),
        bound_offset=TASK6_BOUND_OFFSET
    ).fit(train)
    return write_predictions(TASK6_TEST_FILE, model.predict(test), out_dir, data_dir)
