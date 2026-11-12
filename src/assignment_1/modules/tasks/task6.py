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
  - co-readers of the next book: for each reader v of one of the user's last 5 books a who also
    read the next book b, the implied next time t_ua + (t_vb − t_va). Count, share after t, and
    quartiles of the implied gap (log days). Catches series and books read back to back.
Five regressors with different seeds (early-stopping split), 127 leaves each; log predictions
averaged, then × 1.07. The scale corrects a drift: the model learns from the gap one step before
the one it predicts, and held-out gaps lengthen toward the end of a history (median 43.9 days on
backtest B, 49.1 on A; the test is one step later), so unscaled predictions are too short.
The first-seen bound (first_seen(item) − t_last(user)) is a feature only: applied to the output
as a hard floor (+ 35 days) it raises the error (7,182,796 -> 7,480,674).

Validation
----------
Each user's last training gap held out, predicted from the earlier history
(notebooks/task6.ipynb):
  per-user median 12,052,136 -> shrunk median, last 10 gaps (λ 1.5) 10,137,223 (−15.9%)
  -> + first-seen bound and 35-day offset 8,531,601
  -> GapRegressor 8,012,169 (−33.5% vs the per-user median)
  -> + the next book's reader times 7,182,796 (−10.4%; one step further back in time,
     ≈8.01M -> 7,191,137, −10.2%)
  -> + co-read delay (last 5 books) and 5 seeds 7,110,402 (−1.0%; one step back 7,096,961,
     −1.3%) (notebooks/task6_improve.ipynb, task6_coread.ipynb). Seed-to-seed noise is ~0.3%,
     so k = 1 / 3 / 5 / 10 (−0.9% to −1.1%) are indistinguishable; other readers' gap into the
     book and extra training targets per user gave nothing.
  -> 127 leaves and × 1.07 (best scale 1.05 on B, 1.07 on A): −0.73% on A, −0.48% on B
     against the same 3 seeds (notebooks/task6_tune.ipynb, task6_scale.ipynb). Other settings
     (learning rate, leaf size, L2, max_features, no early stopping) moved < 0.1%.
  Round 5 (notebooks/task6_round5.ipynb), not shipped: cosine similarity of the next book to the
  user's last reads −0.20% (A) / −0.41% (B), + a cap at the data end −0.22% / −0.42%; one scale
  per predicted-gap bucket won on A, lost on B. Error analysis: gaps over 90 days are 37% of
  rows and 80% of the error, predicted at ~30–45% of their length (lapses the features cannot
  foresee), while calibration by predicted gap is already within ±0.16 in log space.
Public: 13,013,427 -> 11,209,411 (bound + offset) -> 10.65M (GapRegressor) -> 8.76M -> 8.64M (co-read, −1.4%)
  -> 8.54M (127 leaves, × 1.07; −1.1%).
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import GapModel, GapRegressor
from assignment_1.modules.util import load_csv, write_predictions

TASK6_GAP_PARAMS = {"k": 10, "lam": 1.5}
# co-read delay over the last 5 books, 5 seeds, 127 leaves, predictions × 1.07 (plan_t6.md)
TASK6_REGRESSOR_PARAMS = {
    "coread": True, "coread_k": 5, "n_bag": 5, "scale": 1.07,
    "regressor": HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.03, max_iter=500, min_samples_leaf=40,
        max_leaf_nodes=127, random_state=0),
}

TASK6_TRAIN_FILE = "task_6_train.csv"
TASK6_TEST_FILE = "task_6_test.csv"


def run_task6(data_dir=None, out_dir="dist"):
    train = load_csv(TASK6_TRAIN_FILE, data_dir)
    test = load_csv(TASK6_TEST_FILE, data_dir)

    model = GapRegressor(base=GapModel(**TASK6_GAP_PARAMS), **TASK6_REGRESSOR_PARAMS).fit(train)
    return write_predictions(TASK6_TEST_FILE, model.predict(test), out_dir, data_dir)
