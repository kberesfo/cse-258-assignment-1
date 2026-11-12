"""Model families. Rating models (Tasks 1, 3) in rating/, ranking models and rules (Tasks 2, 4, 5) in ranking/."""

from .ranking.bpr import BPRModel
from .ranking.ease import EASE, ease_similarity
from .ranking.local import LocalSimilarity
from .ranking.markov import TransitionModel
from .ranking.pairwise import PairFeatures
from .ranking.rules import (
    apply_rules,
    not_yet_seen_rule,
    seen_by_user_rule,
    transition_rule,
    trend_rule,
    unseen_item_rule,
)
from .rating.base import RatingModel
from .rating.bias import BiasModel
from .rating.knn import ItemKNN, TopKItemKNN, UserKNN
from .rating.lfm import SVDModel
from .rating.session import SequenceResidual, SessionResidual
from .rating.stacking import FeatureSource, ModelPrediction, RatingProfile, RatingStats, StackedRating
from .rating.time_bias import TimeBiasModel
from .similarity import (
    ItemSimilarity,
    Similarity,
    adjusted_cosine,
    binary_cosine,
    jaccard,
)
from .time.gaps import GapModel, GapRegressor, split_last, SECONDS_PER_DAY

__all__ = [
    # rating
    "BiasModel",
    "SVDModel",
    "fit_svd",
    "TimeBiasModel",
    "ItemKNN",
    "TopKItemKNN",
    "UserKNN",
    "RatingProfile",
    "SequenceResidual",
    "FeatureSource",
    "RatingStats",
    "StackedRating",
    "ModelPrediction",
    "RatingModel",
    "SessionResidual",
    # ranking
    "BPRModel",
    "EASE",
    "ease_similarity",
    "LocalSimilarity",
    "TransitionModel",
    "PairFeatures",
    # rules
    "apply_rules",
    "not_yet_seen_rule",
    "trend_rule",
    "transition_rule",
    "unseen_item_rule",
    "seen_by_user_rule",
    # time
    "GapModel",
    "GapRegressor",
    "split_last",
    "SECONDS_PER_DAY",
    # similarity
    "ItemSimilarity",
    "Similarity",
    "adjusted_cosine",
    "jaccard",
    "binary_cosine"
]
