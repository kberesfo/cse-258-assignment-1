"""
Task 2: interaction prediction — which of (item1, item2) did the user interact with? Metric: accuracy.

Model: Bayesian Personalized Ranking (implicit BPR: item bias + user/item latent factors),
trained on ALL of tasks_1_2_train.csv; predict the item with the higher score (ties -> item1).

Test design: decoys are popularity-matched (both item columns have the same popularity
distribution as real interactions), so popularity alone scores ~50%; the signal is
personalized taste (gamma_u . gamma_i).

Validation: held-out 10% of interactions, each paired with a popularity-sampled decoy the
user never interacted with, positions randomized (notebooks/task2.ipynb).
Baseline config (5 factors) 0.877 -> chosen config 0.912 (+4.0%).
"""

import numpy as np

from assignment_1.modules.models import bpr_scorer, fit_bpr
from assignment_1.modules.util import load_csv, write_predictions

TASK2_BIAS_PARAMS = {
    "factors": 100,
    "regularization": 0.003,
    "learning_rate": 0.01,
    "iterations": 100,
}


def run_task2(data_dir=None, out_dir="dist"):
    train = load_csv("tasks_1_2_train.csv", data_dir)
    test = load_csv("task_2_test.csv", data_dir)
    fit = fit_bpr(train, **TASK2_BIAS_PARAMS)
    score = bpr_scorer(fit)
    # one score per row for item1
    s1 = score(test["user"].to_numpy(), test["item1"].to_numpy())
    # one score per row for item2
    s2 = score(test["user"].to_numpy(), test["item2"].to_numpy())
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

    return write_predictions("task_2_test.csv", preds, out_dir, data_dir)
