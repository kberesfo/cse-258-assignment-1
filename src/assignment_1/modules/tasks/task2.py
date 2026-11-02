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

from assignment_1.modules.models import bpr_scorer, fit_bpr
from assignment_1.modules.util import load_csv, predict_pairs, write_predictions

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
    preds = predict_pairs(test, bpr_scorer(fit_bpr(train, **TASK2_BPR_PARAMS)))

    return write_predictions(TASK2_TEST_FILE, preds, out_dir, data_dir)
