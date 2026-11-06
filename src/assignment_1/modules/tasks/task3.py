"""
Task 3: rating prediction for (user, item, timestamp). Metric: MSE (rating-0 rows skipped).

Model: bias-only model alpha + beta_u + beta_i (Surprise BaselineOnly, ALS) with separate
user/item regularization (reg_u=3, reg_i=5), trained on the NONZERO ratings of
tasks_3_4_train.csv. The leaderboard skips rows whose true rating is 0 ("shelved, not rated"),
so those rows are dropped from training and validation (the HW3 submission kept them).
Timestamp not yet used.

Validation (notebooks/task3.ipynb; random 75/25 split, scored on nonzero rows; test timestamps
span the same period as train)
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from assignment_1.modules.models import (
    ItemKNN,
    RatingStats,
    SessionResidual,
    StackedRating,
    TimeBiasModel,
)
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK3_BIAS_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "n_epochs": 25}
TASK3_TRAIN_FILE = "tasks_3_4_train.csv"
TASK3_TEST_FILE = "task_3_test.csv"
TASK3_DROP_ZEROS = True


TASK3_TIME_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "reg_ib": 50.0, "n_bins": 10}


def run_task3(data_dir=None, out_dir="dist"):
    train = drop_zero_ratings(load_csv(TASK3_TRAIN_FILE, data_dir))
    test = load_csv(TASK3_TEST_FILE, data_dir)

    model = StackedRating(
        base=TimeBiasModel(**TASK3_TIME_PARAMS),
        sources=[ItemKNN(), RatingStats(), SessionResidual()],
        regressor=HistGradientBoostingRegressor(random_state=0),
    ).fit(train)
    
    return write_predictions(TASK3_TEST_FILE, model.predict(test), out_dir, data_dir)
