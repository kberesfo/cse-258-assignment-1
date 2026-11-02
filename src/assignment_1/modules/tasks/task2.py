"""
Task 2: interaction prediction — which of (item1, item2) did the user interact with? Metric: accuracy.

Model: Bayesian Personalized Ranking (implicit BPR: item bias + user/item latent factors),
trained on ALL of tasks_1_2_train.csv; predict the item with the higher score (ties -> item1).

Test design: decoys are popularity-matched (both item columns have the same popularity
distribution as real interactions), so popularity alone scores ~50%; the signal is
personalized taste (gamma_u . gamma_i).

Validation: held-out 10% of interactions, each paired with a popularity-sampled decoy the
user never interacted with, positions randomized (notebooks/task2.ipynb).
Baseline config (5 factors) 0.877 -> chosen config (100 factors, reg 0.003) 0.915 (+4.3%)
"""

import numpy as np

from assignment_1.modules.models import bpr_scorer, fit_bpr
from assignment_1.modules.util import load_csv, write_predictions

TASK2_BPR_PARAMS = {
    "factors": 100,
    "regularization": 0.003,
    "learning_rate": 0.01,
    "iterations": 100,
}
TASK2_TRAIN_FILE = "tasks_1_2_train.csv"
TASK2_TEST_FILE = "task_2_test.csv"


def run_task2(data_dir=None, out_dir="dist"):
    train = load_csv(TASK2_TRAIN_FILE, data_dir)
    test = load_csv(TASK2_TEST_FILE, data_dir)
    fit = fit_bpr(train, **TASK2_BPR_PARAMS)
    score = bpr_scorer(fit)
    users = test["user"].to_numpy()
    # one score per row for item1
    s1 = score(users, test["item1"].to_numpy())
    # one score per row for item2
    s2 = score(users, test["item2"].to_numpy())
    # chosen item per row; ties -> item1
    preds = np.where(s1 >= s2, test["item1"], test["item2"])

    margin = np.abs(s1 - s2)
    print(
        f"picked item1:   {(preds == test['item1'].to_numpy()).mean():.3f}   (≈0.5 expected)"
    )
    print(f"ties:           {(s1 == s2).mean():.3%}   (≈0 expected)")
    print(
        f"score margin:   median {np.median(margin):.3f}, "
        f"{(margin < 0.1).mean():.1%} of pairs within 0.1"
    )

    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
