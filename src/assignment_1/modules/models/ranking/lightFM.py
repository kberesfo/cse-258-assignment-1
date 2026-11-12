import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset


class LightFMModel:
    """BPR/WARP factorization machine with optional user and item side features (LightFM)."""

    def __init__(self, no_components=32, epochs=30, item_features=None, user_features=None,
                 loss="bpr", seed=0, threads=8, **params):
        """item_features / user_features: callable train -> {id: [feature names]},
        e.g. "pop:3", "year:2012". Identity features are kept, so these add to the ids."""
        self.k, self.epochs, self.loss = no_components, epochs, loss
        self.item_features, self.user_features = item_features, user_features
        self.seed, self.threads, self.params = seed, threads, params

    def fit(self, train):
        ifeats = self.item_features(train) if self.item_features else None
        ufeats = self.user_features(train) if self.user_features else None
        def names(d):
            return {f for fs in d.values() for f in fs} if d else None

        ds = Dataset()
        ds.fit(train["user"].unique(), train["item"].unique(),
               user_features=names(ufeats), item_features=names(ifeats))
        inter, _ = ds.build_interactions(zip(train["user"], train["item"]))
        self.IF_ = ds.build_item_features(ifeats.items()) if ifeats else None
        self.UF_ = ds.build_user_features(ufeats.items()) if ufeats else None
        self.uid_, _, self.iid_, _ = ds.mapping()
        self.model_ = LightFM(loss=self.loss, no_components=self.k, random_state=self.seed,
                              **self.params)
        self.model_.fit(inter, user_features=self.UF_, item_features=self.IF_,
                        epochs=self.epochs, num_threads=self.threads)
        return self

    def score(self, users, items):
        """Score per (user, item); NaN where either is unseen in training."""
        u = np.array([self.uid_.get(x, -1) for x in users], dtype=np.int32)
        i = np.array([self.iid_.get(x, -1) for x in items], dtype=np.int32)
        out, ok = np.full(len(u), np.nan), (u >= 0) & (i >= 0)
        out[ok] = self.model_.predict(u[ok], i[ok], user_features=self.UF_,
                                      item_features=self.IF_, num_threads=self.threads)
        return out

    __call__ = score
