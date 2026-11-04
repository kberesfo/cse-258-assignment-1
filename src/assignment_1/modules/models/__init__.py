"""Model families. Rating models (Tasks 1, 3) in rating/, ranking models and rules (Tasks 2, 4, 5) in ranking/."""

from .ranking.bpr import BPRFit, bpr_scorer, fit_bpr
from .ranking.markov import TransitionFit, fit_transitions, transition_scorer
from .ranking.rules import (
    apply_rules,
    not_yet_seen_rule,
    transition_rule,
    trend_rule,
    unseen_item_rule,
)
from .rating.bias import fit_bias, predict_bias
from .rating.lfm import fit_svd, predict_svd, to_surprise_trainset
from .rating.time_bias import TimeBiasFit, fit_time_bias, predict_time_bias
from .time.gaps import GapFit, fit_gap_model, item_first_seen_bound, predict_gaps
from .ranking.pairwise import cand_features, pair_features, PairContext, fit_pair_context
from .ranking.similarity import CosineFit, fit_item_cosine, cos_features

__all__ = [
    # rating
    "fit_bias",
    "predict_bias",
    "fit_svd",
    "predict_svd",
    "to_surprise_trainset",
    # ranking
    "BPRFit",
    "TimeBiasFit",
    "bpr_scorer",
    "fit_bpr",
    "TransitionFit",
    "fit_transitions",
    "transition_scorer",
    "fit_time_bias",
    "predict_time_bias",
    "cand_features",
    "pair_features",
    "PairContext",
    "fit_pair_context"
    # rules
    "apply_rules",
    "not_yet_seen_rule",
    "trend_rule",
    "transition_rule",
    "unseen_item_rule",
    # time
    "GapFit",
    "fit_gap_model",
    "predict_gaps",
    "item_first_seen_bound",
    # similarity
    "CosineFit",
    "fit_item_cosine",
    "cos_features"
]
