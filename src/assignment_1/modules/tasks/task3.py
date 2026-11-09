"""
Task 3: rating prediction for (user, item, timestamp). Metric: MSE (lower is better);
rows whose true rating is 0 ("shelved, not rated") are skipped by the leaderboard.

Test design
-----------
Test rows carry timestamps and are drawn like a random split: 94.7% fall inside the user's
training time span (2.6% after it), and test timestamp quantiles match train's. So a random
validation split is appropriate, and features may use the user's training rows both before and
after the row's time. ~0.15% of timestamps fall outside 2005–2018 and are clipped.
0-ratings are dropped from training and validation (the HW3 submission kept them).

Model
-----
StackedRating: a base rating model plus a gradient-boosted regressor trained on the base
model's residuals (an ensemble of standard models over user/item features, as in lecture).
  Base: TimeBiasModel, α + β_u + β_i + β_{i,bin(t)} fit by ALS (reg_u 3, reg_i 5, reg_ib 50,
  10 quantile time bins).
  Feature sources (all fitted on training rows only):
    - TopKItemKNN, Jaccard: item-kNN on base residuals over the whole history, plus over only the
      user's k closest books (k = 1–20), the closest book's rating, and time-weighted means
      (14 / 90 / 365 days)
    - item-kNN on base residuals, adjusted cosine shrunk by co-support (rating agreement)
    - item-kNN on raw ratings, Jaccard similarity (similarity-weighted average rating)
    - UserKNN: residuals of the user's 100 most similar readers (IDF-weighted cosine) on this book
    - RatingStats: user / item history size, user rating std
    - RatingProfile: user / item mean, spread, share of 1–2 and 5-star ratings, activity span
    - SessionResidual: the user's mean residual on training rows from the same day / week
    - SequenceResidual: mean residual of the user's 1 / 3 / 5 ratings just before / after the
      row's time, and the days to each
  Each kNN source gives Σ S(i,j)·v_uj / Σ|S(i,j)| over the history plus its support.
  HistGradientBoostingRegressor (lr 0.03, 800 iterations, min leaf 100, l2 5) is trained
  out-of-fold over 5 folds, so it learns from every row; base and sources are then refit on all
  nonzero rows for prediction.

Validation
----------
Yardsticks on a random 75/25 split (notebooks/task3.ipynb): bias model 0.7449, time-bias 0.7406.
Outer 10% holdout, first stack (single internal split):
  bias only 0.7369 -> + cosine kNN + stats 0.7066 -> time-bias base 0.7024 -> + session 0.7019
  -> Jaccard + adjusted-cosine + raw-rating kNN 0.6908 (−6.3% vs bias only)
Ablation on a new outer 10% holdout (notebooks/task3_ablation.ipynb; GB seed pinned, so the
shipped stack scores 0.6931 here):
  shipped 0.6931 -> tuned GB 0.6926 -> 5 folds 0.6908 -> top-k kNN 0.6833 -> user kNN 0.6812
  -> profile 0.6775 -> sequence 0.6760 (−2.5%)
  Cost of removing each from the full model: top-k +0.0058, profile +0.0038, user kNN +0.0022,
  sequence +0.0015.
  Four outer splits: −0.0163 on average (−0.0153 to −0.0170), better on 4/4.
Public: 0.7921 (with 0s) -> 0.7626 (0s dropped) -> 0.7229 (first stack) -> 0.7084
-> 0.6881 (top-k / user kNN / profile / sequence, 5 folds).
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import (
    ItemKNN,
    RatingProfile,
    RatingStats,
    SequenceResidual,
    SessionResidual,
    StackedRating,
    TimeBiasModel,
    TopKItemKNN,
    UserKNN,
    adjusted_cosine,
    jaccard,
)
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK3_TRAIN_FILE = "tasks_3_4_train.csv"
TASK3_TEST_FILE = "task_3_test.csv"
TASK3_DROP_ZEROS = True

TASK3_TIME_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "reg_ib": 50.0, "n_bins": 10}
TASK3_GB_PARAMS = {"learning_rate": 0.03, "max_iter": 800, "min_samples_leaf": 100,
                   "l2_regularization": 5.0, "random_state": 0}


def run_task3(data_dir=None, out_dir="dist"):
    train = drop_zero_ratings(load_csv(TASK3_TRAIN_FILE, data_dir))
    test = load_csv(TASK3_TEST_FILE, data_dir)

    model = StackedRating(
        base=TimeBiasModel(**TASK3_TIME_PARAMS),
        sources=[
            TopKItemKNN(similarity=jaccard, prefix="knn_jac",
                        time_scales=(14, 90, 365)),
            ItemKNN(similarity=adjusted_cosine, prefix="knn_adj"),
            ItemKNN(similarity=jaccard, target="rating", prefix="knn_raw"),
            UserKNN(),
            RatingStats(),
            RatingProfile(),
            SessionResidual(),
            SequenceResidual(),
        ],
        regressor=HistGradientBoostingRegressor(**TASK3_GB_PARAMS),
        n_folds=5,
        n_jobs=3,
    ).fit(train)

    return write_predictions(TASK3_TEST_FILE, model.predict(test), out_dir, data_dir)
