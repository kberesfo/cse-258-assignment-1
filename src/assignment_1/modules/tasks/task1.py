"""
Task 1: rating prediction for (user, item) pairs, no timestamp. Metric: MSE (lower is better);
rows whose true rating is 0 ("shelved, not rated") are skipped by the leaderboard.

Test design
-----------
No test timestamps; 0% of test (user, item) pairs appear in train (no re-read leak). 1.48% of
training rows are 0-ratings; they are dropped from training and validation (the HW3 submission
kept them). The bias model's error is concentrated in low ratings: 1–2 star rows are 7% of
validation but 36% of squared error, i.e. user-specific dislikes that β_u cannot express.

Model
-----
StackedRating: a base rating model plus a gradient-boosted regressor trained on the base
model's residuals (an ensemble of standard models over user/item features, as in lecture).
  Base: BiasModel, α + β_u + β_i fit by ALS (Surprise BaselineOnly; reg_u 1, reg_i 5, 25 epochs).
  Feature sources (all fitted on training rows only):
    - TopKItemKNN, Jaccard: item-kNN on base residuals over the whole history, plus over only the
      user's k closest books (k = 1–20) and the closest book's rating; a whole-history average
      dilutes the few close books (e.g. a series)
    - item-kNN on base residuals, adjusted cosine shrunk by co-support (rating agreement)
    - item-kNN on raw ratings, Jaccard similarity (similarity-weighted average rating)
    - UserKNN: residuals of the user's 100 most similar readers (IDF-weighted cosine) on this book
    - RatingStats: user / item history size, user rating std
    - RatingProfile: user / item mean, spread, share of 1–2 and 5-star ratings, activity span
  Each kNN source gives Σ S(i,j)·v_uj / Σ|S(i,j)| over the history plus its support.
  HistGradientBoostingRegressor (lr 0.03, 800 iterations, min leaf 100, l2 5) is trained
  out-of-fold over 5 folds, so it learns from every row; base and sources are then refit on all
  nonzero rows for prediction.

Validation
----------
Random 75/25 split (notebooks/task1.ipynb): bias model 0.5941. Latent factors never beat it:
SVD with one λ 0.5989; separate bias / factor λ 0.5980; converged by ~120 epochs (~0.597).
Item-kNN on bias residuals, hand-tuned (b̂ + a·num / (den + λ)): 0.5674; stacked on a held-out
half: 0.5614.
Outer 10% holdout, first stack (single internal split):
  bias only 0.5876 -> stack with cosine kNN + stats 0.5514 -> Jaccard 0.5492
  -> Jaccard + adjusted cosine 0.5465 -> + raw-rating kNN 0.5454 (−7.2% vs bias only)
  Tried without gain: adjusted cosine alone (0.5640, noisy), SVD K=1 / K=5 as extra
  ensemble members (redundant with kNN).
Ablation on a new outer 10% holdout (notebooks/task1_ablation.ipynb; GB seed pinned, so the
shipped stack scores 0.5430 here):
  shipped 0.5430 -> tuned GB 0.5429 -> 5 folds 0.5414 -> top-k kNN 0.5352 -> user kNN 0.5332
  -> profile 0.5299 (−2.4%)
  Cost of removing each from the full model: top-k +<x>, user kNN +<x>, profile +0.0033.
  Four outer splits: −<mean diff> on average, better on <n>/4.
Public: 0.6185 (with 0s) -> 0.5917 (0s dropped) -> 0.5501 (first stack) -> 0.5453
(Jaccard + adjusted) -> 0.5443 (+ raw-rating kNN) -> 0.5296 (top-k / user kNN / profile, 5 folds).
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import (
    BiasModel,
    ItemKNN,
    RatingProfile,
    RatingStats,
    StackedRating,
    TopKItemKNN,
    UserKNN,
    adjusted_cosine,
    jaccard,
)
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK1_BIAS_PARAMS = {"reg_u": 1.0, "reg_i": 5.0, "n_epochs": 25}
TASK1_GB_PARAMS = {"learning_rate": 0.03, "max_iter": 800, "min_samples_leaf": 100,
                   "l2_regularization": 5.0, "random_state": 0}
TASK1_TRAIN_FILE = "tasks_1_2_train.csv"
TASK1_TEST_FILE = "task_1_test.csv"
TASK1_DROP_ZEROS = True


def run_task1(data_dir=None, out_dir="dist"):
    train = drop_zero_ratings(load_csv(TASK1_TRAIN_FILE, data_dir))
    test = load_csv(TASK1_TEST_FILE, data_dir)

    model = StackedRating(
        base=BiasModel(**TASK1_BIAS_PARAMS),
        sources=[
            TopKItemKNN(similarity=jaccard, prefix="knn_jac"),
            ItemKNN(similarity=adjusted_cosine, prefix="knn_adj"),
            ItemKNN(similarity=jaccard, target="rating", prefix="knn_raw"),
            UserKNN(),
            RatingStats(),
            RatingProfile(),
        ],
        regressor=HistGradientBoostingRegressor(**TASK1_GB_PARAMS),
        n_folds=5,
        n_jobs=3,
    ).fit(train)

    return write_predictions(TASK1_TEST_FILE, model.predict(test), out_dir, data_dir)
