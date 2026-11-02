"""
Task 5: next-item prediction — which of(item1, item2) is the user's next interaction after
their training history? Metric: accuracy. Test has no timestamp; only history order is usable.

Model: BPR (implicit; item bias + user/item latent factors) trained on ALL of
task_5_train.csv; predict the item with the higher score (ties -> item1).

Rule: if exactly one candidate never appears in training, predict it. Decoys are drawn from
training interactions (popularity-matched, so never unseen), so an unseen candidate must be the
real next item — a book nobody else read. Applies to ~3.96% of test rows (never both unseen);
consistent with ~3.8% of users whose held-out last item is unseen in validation.

Test design: one row per user (86,786); decoys popularity-matched (medians ~100 vs 91
for interactions), so popularity alone scores ~50%.

Validation: each user's most recent interaction held out (Holdout.LAST), paired with a
popularity-sampled decoy; rows with an unseen held-out item are dropped, so validation measures
the ~96% of rows the model decides (notebooks/task5.ipynb).
Baseline config (5 factors) 0.8425 ->  tuned BPR (50 factors, reg 0.005) 0.8704 (+3.3%) -> + transition rule (min gap 5) 0.8731 (+3.6%).
"""


from assignment_1.modules.models import (
    apply_rules,
    bpr_scorer,
    fit_bpr,
    fit_transitions,
    transition_rule,
    unseen_item_rule,
)
from assignment_1.modules.util import load_csv, predict_pairs, write_predictions

TASK5_BPR_PARAMS = {
    "factors": 50,
    "regularization": 0.005,
    "learning_rate": 0.01,
    "iterations": 100
}
TASK5_MIN_GAP = 5
TASK5_TRAIN_FILE = "task_5_train.csv"
TASK5_TEST_FILE = "task_5_test.csv"


def run_task5(data_dir=None, out_dir="dist"):
    train = load_csv(TASK5_TRAIN_FILE, data_dir)
    test = load_csv(TASK5_TEST_FILE, data_dir)

    preds = predict_pairs(
        test,
        bpr_scorer(
            fit_bpr(train, **TASK5_BPR_PARAMS)
        )
    )
    preds = apply_rules(
        preds,
        # strong transition evidence
        transition_rule(fit_transitions(train), test, min_gap=TASK5_MIN_GAP),
        # applied last: wins any conflict
        unseen_item_rule(train, test)
    )

    return write_predictions(TASK5_TEST_FILE, preds, out_dir, data_dir)
