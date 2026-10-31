"""Task 1: rating prediction for (user, item) pairs, no timestamp. Metric: MSE.

Model: latent factor model (Surprise SVD), trained on ALL of tasks_1_2_train.csv.
Hyperparameters chosen on a 75/25 validation split in notebooks/task1.ipynb.
"""

from assignment_1.models.lfm import fit_svd, predict_svd
from assignment_1.modules.util import load_csv, write_predictions

# TODO: replace with the best settings from notebooks/task1.ipynb.
# Currently the baseline's settings (but retrained on 100% of the data, unlike baselines.py).
TASK1_SVD_PARAMS = {"n_factors": 1, "n_epochs": 50}


def run_task1(data_dir=None, out_dir="dist"):
    train = load_csv("tasks_1_2_train.csv", data_dir)
    test = load_csv("task_1_test.csv", data_dir)
    model = fit_svd(train, **TASK1_SVD_PARAMS)
    preds = predict_svd(model, test["user"], test["item"])

    return write_predictions("task_1_test.csv", preds, out_dir, data_dir)
