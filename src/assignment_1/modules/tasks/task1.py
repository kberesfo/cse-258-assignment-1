"""
Task 1: rating prediction for (user, item) pairs, no timestamp. Metric: MSE.

Model: bias-only model alpha + beta_u + beta_i (Surprise BaselineOnly, ALS), separate user/item
regularization, trained on ALL of tasks_1_2_train.csv. Chosen in notebooks/task1.ipynb:
validation MSE 0.7698 vs 0.7863 for the baseline SVD config (+2.1%). Latent factors (SVD,
1-50 factors) did not improve on the biases.
"""

import numpy as np

from assignment_1.modules.models import fit_bias, predict_bias
from assignment_1.modules.util import load_csv, write_predictions

TASK1_BIAS_PARAMS = {"reg_u": 1.0, "reg_i": 5.0, "n_epochs": 25}
TASK_TRAIN_FILE = "tasks_1_2_train.csv"
TASK_TEST_FILE = "task_1_test.csv"


def run_task1(data_dir=None, out_dir="dist"):
    train = load_csv(TASK_TRAIN_FILE, data_dir)
    test = load_csv(TASK_TEST_FILE, data_dir)
    model = fit_bias(train, **TASK1_BIAS_PARAMS)
    preds = predict_bias(model, test["user"], test["item"])

    preds = np.asarray(preds)

    return write_predictions(TASK_TEST_FILE, preds, out_dir, data_dir)
