"""
Task 1: 
rating prediction for (user, item) pairs, no timestamp.
Metric: MSE (rating-0 rows skipped).

Model: bias-only model alpha + beta_u + beta_i (Surprise BaselineOnly, ALS) with separate
user/item regularization (reg_u=1, reg_i=5), trained on the NONZERO ratings of
tasks_1_2_train.csv. The leaderboard skips rows whose true rating is 0 ("shelved, not rated"),
so those rows are dropped from training and validation (the HW3 submission kept them).

Validation (notebooks/task1.ipynb; 75/25 split, scored on nonzero rows): bias model 0.594;
SVD (1 factor, reg_all 0.05, 50 epochs) 0.599; bias+SVD blend no gain (0.594). Latent factors
did not improve on the biases.
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import (
    BiasModel,
    ItemKNN,
    RatingStats,
    StackedRating,
    adjusted_cosine,
    jaccard,
)
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK1_BIAS_PARAMS = {"reg_u": 1.0, "reg_i": 5.0, "n_epochs": 25}
TASK1_TRAIN_FILE = "tasks_1_2_train.csv"
TASK1_TEST_FILE = "task_1_test.csv"
TASK1_DROP_ZEROS = True


def run_task1(data_dir=None, out_dir="dist"):
    train = drop_zero_ratings(load_csv(TASK1_TRAIN_FILE, data_dir))
    test = load_csv(TASK1_TEST_FILE, data_dir)

    model = StackedRating(
        base=BiasModel(**TASK1_BIAS_PARAMS),
        sources=[
            ItemKNN(similarity=jaccard, prefix="knn_jac"),
            ItemKNN(similarity=adjusted_cosine, prefix="knn_adj"),
            RatingStats(),
        ],
        regressor=HistGradientBoostingRegressor(random_state=0),
    ).fit(train)

    return write_predictions(TASK1_TEST_FILE, model.predict(test), out_dir, data_dir)
