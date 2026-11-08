"""
Task 3: rating prediction for (user, item, timestamp). Metric: MSE (lower is better);
rows whose true rating is 0 ("shelved, not rated") are skipped by the leaderboard.

Test design
-----------
Test rows carry timestamps and are drawn like a random split: 94.7% fall inside the user's
training time span (2.6% after it), and test timestamp quantiles match train's. So a random
validation split is appropriate. ~0.15% of timestamps fall outside 2005–2018 and are clipped.
0-ratings are dropped from training and validation (the HW3 submission kept them).

Model
-----
StackedRating: a base rating model plus a gradient-boosted regressor trained on the base
model's residuals (an ensemble of standard models over user/item features, as in lecture).
  Base: TimeBiasModel, α + β_u + β_i + β_{i,bin(t)} fit by ALS (reg_u 3, reg_i 5, reg_ib 50,
  10 quantile time bins).
  Feature sources (all fitted on training rows only):
    - item-kNN on base residuals, Jaccard similarity (co-reading)
    - item-kNN on base residuals, adjusted cosine shrunk by co-support (rating agreement)
    - item-kNN on raw ratings, Jaccard similarity (similarity-weighted average rating)
    - RatingStats: user / item history size, user rating std
    - SessionResidual: the user's mean residual on training rows from the same day / week
  Each kNN source gives Σ S(i,j)·v_uj / Σ|S(i,j)| over the user's history plus its support.
  HistGradientBoostingRegressor (defaults) learns the correction on an internal 75/25 split;
  base and sources are then refit on all nonzero rows for prediction.

Validation
----------
Yardsticks on a random 75/25 split (notebooks/task3.ipynb): bias model 0.7449, time-bias 0.7406.
Outer 10% holdout (each stack trains its regressor on its own internal split):
  bias only 0.7369 -> + cosine kNN + stats 0.7066 -> time-bias base 0.7024 -> + session 0.7019
  -> Jaccard + adjusted-cosine + raw-rating kNN 0.6908 (−6.3% vs bias only)
Public: 0.7921 (with 0s) -> 0.7626 (0s dropped) -> 0.7229 (first stack) -> <after upload>.
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import (
    ItemKNN,
    RatingStats,
    SessionResidual,
    StackedRating,
    TimeBiasModel,
    adjusted_cosine,
    jaccard,
)
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK3_TRAIN_FILE = "tasks_3_4_train.csv"
TASK3_TEST_FILE = "task_3_test.csv"
TASK3_DROP_ZEROS = True

TASK3_TIME_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "reg_ib": 50.0, "n_bins": 10}


def run_task3(data_dir=None, out_dir="dist"):
    train = drop_zero_ratings(load_csv(TASK3_TRAIN_FILE, data_dir))
    test = load_csv(TASK3_TEST_FILE, data_dir)

    model = StackedRating(
        base=TimeBiasModel(**TASK3_TIME_PARAMS),
        sources=[
            ItemKNN(similarity=jaccard, prefix="knn_jac"),
            ItemKNN(similarity=adjusted_cosine, prefix="knn_adj"),
            ItemKNN(similarity=jaccard, target="rating", prefix="knn_raw"),
            RatingStats(),
            SessionResidual(),
        ],
        regressor=HistGradientBoostingRegressor(random_state=0),
    ).fit(train)

    return write_predictions(TASK3_TEST_FILE, model.predict(test), out_dir, data_dir)
