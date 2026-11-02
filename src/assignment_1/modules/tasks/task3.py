"""
Task 3: rating prediction for (user, item, timestamp). Metric: MSE.

Model: bias-only model alpha + beta_u + beta_i (Surprise BaselineOnly, ALS), separate user/item
regularization, trained on ALL of tasks_3_4_train.csv; timestamp not yet used.
Chosen in notebooks/task3.ipynb (random 75/25 split; test timestamps span the same period as
train): validation MSE 0.9229 vs 0.9428 for the baseline SVD config (+2.1%).
"""

import numpy as np

from assignment_1.modules.models import fit_bias, predict_bias
from assignment_1.modules.util import load_csv, write_predictions

TASK3_BIAS_PARAMS = {"reg_u": 3.0, "reg_i": 5.0, "n_epochs": 25}
TASK3_TRAIN_FILE = "tasks_3_4_train.csv"
TASK3_TEST_FILE = "task_3_test.csv"


def run_task3(data_dir=None, out_dir="dist"):
    train = load_csv(TASK3_TRAIN_FILE, data_dir)
    test = load_csv(TASK3_TEST_FILE, data_dir)
    model = fit_bias(train, **TASK3_BIAS_PARAMS)
    preds = predict_bias(model, test["user"], test["item"])

    print(f"pred mean {preds.mean():.3f} (train mean {train.rating.mean():.3f})")
    print(f"pred range [{preds.min():.2f}, {preds.max():.2f}] (allowed 0–5)")
    print(f"pred std {preds.std():.3f} (train std {train.rating.std():.3f})")

    return write_predictions(TASK3_TEST_FILE, preds, out_dir, data_dir)
