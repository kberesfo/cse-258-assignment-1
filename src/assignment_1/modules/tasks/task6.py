"""
Task 6: time-delta prediction. How long (seconds) until the user's next interaction?
Metric: MAE (lower is better).

Test design
-----------
One row per user, with the item of the next interaction. Metric confirmed as MAE (on validation
the baseline's MAE matched the public baseline in scale; its MSE was orders of magnitude off).
Gaps are heavily skewed (median ~13 days, mean ~71 days), so medians, not means: MAE is
minimized by the median. The test gives the next item, so when other readers read it is
usable information: the book can't be read before it first appears.

Model
-----
GapRegressor (time-series regression on the previous gaps, as suggested in lecture):
HistGradientBoostingRegressor with MAE loss on log1p(gap). Since median(log g) = log(median g),
fitting MAE in log space still estimates the median gap. Trained by stepping back one
interaction per user (each user's last training gap is the target, earlier history gives the
features). Features:
  - the last 8 log gaps; log-gap median / mean / std / count / quartiles / max; medians of the
    last 3, 5, 10 and 20 gaps; recent activity (30 / 90 / 365 days); calendar position, time left
    before the data ends; last and mean rating
  - GapModel's prediction: per-user median of the last 10 gaps, shrunk toward the global median
    in log space, log ĝ = w log m_u + (1 − w) log m, w = n_u / (n_u + 1.5)
  - when other readers read the next book, relative to the user's last interaction t: first and
    last reader, how many read it after t, quantiles of those times, readers in the next
    30 / 90 / 365 days
The first-seen bound (first_seen(item) − t_last(user)) is a feature only: applied to the output
as a hard floor (+ 35 days) it raises the error (<7.18M> -> <7.48M>).

Validation
----------
Each user's last training gap held out, predicted from the earlier history
(notebooks/task6.ipynb):
  per-user median 12,052,136 -> shrunk median, last 10 gaps (λ 1.5) 10,137,223 (−15.9%)
  -> + first-seen bound and 35-day offset 8,531,601
  -> GapRegressor 8,012,169 (−33.5% vs the per-user median)
  -> + the next book's reader times <7,183,xxx> (−10.4%; one step further back in time,
     <8,009,xxx> -> <7,191,xxx>, −10.2%)
Public: 13,013,427 -> 11,209,411 (bound + offset) -> 10.65M (GapRegressor) -> 0.6881 (top-k / user kNN / profile / sequence, 5 folds).
"""

from assignment_1.modules.models import GapModel, GapRegressor
from assignment_1.modules.util import load_csv, write_predictions

TASK6_GAP_PARAMS = {"k": 10, "lam": 1.5}

TASK6_TRAIN_FILE = "task_6_train.csv"
TASK6_TEST_FILE = "task_6_test.csv"


def run_task6(data_dir=None, out_dir="dist"):
    train = load_csv(TASK6_TRAIN_FILE, data_dir)
    test = load_csv(TASK6_TEST_FILE, data_dir)

    model = GapRegressor(base=GapModel(**TASK6_GAP_PARAMS)).fit(train)
    return write_predictions(TASK6_TEST_FILE, model.predict(test), out_dir, data_dir)
