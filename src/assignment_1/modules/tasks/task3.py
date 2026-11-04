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

from assignment_1.modules.models import fit_bias, predict_bias
from assignment_1.modules.util import drop_zero_ratings, load_csv, write_predictions

TASK3_BIAS_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "n_epochs": 25}
TASK3_TRAIN_FILE = "tasks_3_4_train.csv"
TASK3_TEST_FILE = "task_3_test.csv"
TASK3_DROP_ZEROS = True


def run_task3(data_dir=None, out_dir="dist"):
    train = load_csv(TASK3_TRAIN_FILE, data_dir)
    test = load_csv(TASK3_TEST_FILE, data_dir)

    if TASK3_DROP_ZEROS:
        train = drop_zero_ratings(train)

    model = fit_bias(train, **TASK3_BIAS_PARAMS)
    preds = predict_bias(model, test["user"], test["item"])

    return write_predictions(TASK3_TEST_FILE, preds, out_dir, data_dir)
