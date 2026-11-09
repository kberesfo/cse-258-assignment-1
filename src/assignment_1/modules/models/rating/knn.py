"""Neighbourhood features on a base model's residuals (PML §4.3, similarity-based rating prediction).

ItemKNN      the user's history items, weighted by their similarity to the target item
TopKItemKNN  ItemKNN restricted to the user's closest history items
UserKNN      the target item's readers, weighted by their similarity to the user

All are FeatureSources for StackedRating: fit on the rows the base model was fit on, then give
per-row features. Assumes one row per (user, item): the sparse matrices sum duplicates.
"""

from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from ..similarity import ItemSimilarity, Similarity, binary_cosine
from .base import RatingModel
from .stacking import FeatureSource


class ItemKNN(FeatureSource):
    """
    Item-kNN features for (u, i), relative to a fitted base model b̂.

    With the user's history H_u, item-item similarity S(i, j), and v_uj = r_uj − b̂_uj
    (target="residual") or r_uj (target="rating"):
        num = Σ_{j ∈ H_u} S(i, j) · v_uj
        den = Σ_{j ∈ H_u} |S(i, j)|
    Features:
        {prefix}              residual: num / (den + shrink), a correction shrunk toward 0
                              rating:   num / den, a weighted average rating, NaN if den = 0
                              (shrinking a rating toward 0 would be wrong)
        {prefix}_log_support  log(1 + den)
    Unknown users or items get num = den = 0.
    `similarity` is any (R, Rres) -> S function: binary_cosine, jaccard, adjusted_cosine.
    """

    def __init__(self, similarity: Similarity = binary_cosine, shrink: float = 1.0,
                 prefix: str = "knn", target: str = "residual", chunk: int = 50_000) -> None:
        """target: "residual" averages r − b̂ (default); "rating" averages raw ratings r."""
        if target not in ("residual", "rating"):
            raise ValueError(
                f"target must be 'residual' or 'rating', got {target!r}")
        self.similarity, self.shrink, self.prefix = similarity, shrink, prefix
        self.target, self.chunk = target, chunk
        self.sim: ItemSimilarity | None = None

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        """train: rated rows the base model was fit on; base: a fitted RatingModel."""
        y = train["rating"].to_numpy(dtype=float)
        vals = y - base.predict(train) if self.target == "residual" else y
        self.sim = ItemSimilarity(self.similarity).fit(train, residuals=vals)
        return self

    def parts(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(num, den) per row; 0 where the user or item is unknown."""
        if self.sim is None:
            raise RuntimeError("ItemKNN.parts called before fit")
        sim = self.sim
        u = sim.users.get_indexer(rows["user"])
        i = sim.items.get_indexer(rows["item"])
        num, den = np.zeros(len(u)), np.zeros(len(u))

        for s in range(0, len(u), self.chunk):
            uu, ii = u[s:s + self.chunk], i[s:s + self.chunk]
            ok = (uu >= 0) & (ii >= 0)
            Si = sim.S[ii[ok]]
            num[s:s + self.chunk][ok] = np.asarray(
                sim.Rres[uu[ok]].multiply(Si).sum(axis=1)).ravel()
            den[s:s + self.chunk][ok] = np.asarray(
                abs(sim.R[uu[ok]].multiply(Si)).sum(axis=1)).ravel()
        return num, den

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        num, den = self.parts(rows)
        p = self.prefix
        if self.target == "rating":                    # weighted average rating; NaN if no support
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(den > 0, num / den, np.nan)
        else:                                          # shrunk residual correction
            est = num / (den + self.shrink)
        return pd.DataFrame({p: est, f"{p}_log_support": np.log1p(den)})


class TopKItemKNN(ItemKNN):
    """
    ItemKNN plus features from only the user's most similar history items. Averaging over the
    whole history dilutes the few books that are really close (e.g. the same series).

    With history items j ranked by S(i, j), highest first, and e_uj = r_uj − b̂_uj:
        {prefix}_top{k}     Σ_top-k S·e / (Σ_top-k S + 0.05)
        {prefix}_top{k}_s   Σ_top-k S (how close the top k are)
        {prefix}_nn_rating  raw rating of the single closest book
        {prefix}_nn_sim     its similarity
        {prefix}_sq         Σ S²·e / (Σ S² + 0.01) (sharpens toward close items)
        {prefix}_n          log(1 + number of history items with S ≠ 0)
        {prefix}_tw{τ}      Σ w·e / (Σ w + 0.02), w = S·exp(−|Δt| / τ days); needs row timestamps
    plus ItemKNN's {prefix} and {prefix}_log_support (always on residuals). The means are NaN
    where there is no support.

    Use a non-negative similarity (jaccard, binary_cosine). With adjusted_cosine, negative S
    still enters the sums, and a den near zero makes the means blow up.
    """

    def __init__(self, similarity: Similarity = binary_cosine, prefix: str = "knn",
                 ks: tuple[int, ...] = (1, 3, 5, 10, 20), time_scales: tuple[int, ...] = (),
                 shrink: float = 1.0, chunk: int = 20_000) -> None:
        super().__init__(similarity=similarity, shrink=shrink, prefix=prefix, chunk=chunk)
        self.ks, self.time_scales = ks, time_scales

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        super().fit(train, base)
        sim = self.sim
        self.rating_ = train["rating"].to_numpy(dtype=float)
        self.res_ = self.rating_ - base.predict(train)
        self.times_ = train["timestamp"].to_numpy(dtype=float)
        # history matrix whose entries point back to the training row (offset by 1: 0 is "empty")
        self.row_of_ = csr_matrix(
            (np.arange(1, len(train) + 1, dtype=np.float64),
             (sim.users.get_indexer(train["user"]), sim.items.get_indexer(train["item"]))),
            shape=sim.R.shape)
        return self

    def _entries(self, uu: np.ndarray, ii: np.ndarray):
        """Yield (start, row, training row, similarity) for history items with S > 0, by chunk."""
        for s in range(0, len(uu), self.chunk):
            H, Si = self.row_of_[uu[s:s + self.chunk]
                                 ], self.sim.S[ii[s:s + self.chunk]]
            A = H.multiply(Si.astype(bool)).tocsr()
            B = Si.multiply(H.astype(bool)).tocsr()
            A.sort_indices()
            B.sort_indices()
            row = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
            yield s, row, A.data.astype(np.int64) - 1, B.data

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        p, n_rows = self.prefix, len(rows)
        names = [f"{p}_nn_rating", f"{p}_nn_sim", f"{p}_sq", f"{p}_n"]
        names += [f"{p}_top{k}{x}" for k in self.ks for x in ("", "_s")]
        names += [f"{p}_tw{t}" for t in self.time_scales]
        out = {name: np.full(n_rows, np.nan) for name in names}

        u = self.sim.users.get_indexer(rows["user"])
        i = self.sim.items.get_indexer(rows["item"])
        known = np.flatnonzero((u >= 0) & (i >= 0))
        query = rows["timestamp"].to_numpy(
            dtype=float) if self.time_scales else None

        for start, row, train_row, sim in self._entries(u[known], i[known]):
            where = known[start:start + self.chunk]
            n = len(where)
            order = np.lexsort((-sim, row))
            row, train_row, sim = row[order], train_row[order], sim[order]
            firsts = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
            rank = np.arange(len(row)) - np.repeat(firsts,
                                                   np.diff(np.r_[firsts, len(row)]))
            e = self.res_[train_row]

            for k in self.ks:
                m = rank < k
                num = np.bincount(row[m], sim[m] * e[m], minlength=n)
                den = np.bincount(row[m], sim[m], minlength=n)
                out[f"{p}_top{k}"][where] = np.where(
                    den > 0, num / (den + 0.05), np.nan)
                out[f"{p}_top{k}_s"][where] = den

            top = rank == 0
            out[f"{p}_nn_rating"][where[row[top]]
                                  ] = self.rating_[train_row[top]]
            out[f"{p}_nn_sim"][where[row[top]]] = sim[top]
            num = np.bincount(row, sim ** 2 * e, minlength=n)
            den = np.bincount(row, sim ** 2, minlength=n)
            out[f"{p}_sq"][where] = np.where(
                den > 0, num / (den + 0.01), np.nan)
            out[f"{p}_n"][where] = np.log1p(np.bincount(row, minlength=n))

            if self.time_scales:
                days = np.abs(self.times_[train_row] -
                              query[where[row]]) / 86400
                for t in self.time_scales:
                    w = sim * np.exp(-days / t)
                    num, den = np.bincount(
                        row, w * e, minlength=n), np.bincount(row, w, minlength=n)
                    out[f"{p}_tw{t}"][where] = np.where(
                        den > 0, num / (den + 0.02), np.nan)

        return pd.concat([super().features(rows), pd.DataFrame(out)], axis=1)


class UserKNN(FeatureSource):
    """
    User-kNN features: how the user's nearest readers did on this item, relative to the base model.

    Neighbours N(u): each user's top_n users by IDF-weighted cosine over books read (rare shared
    books count more). Not symmetric: v ∈ N(u) doesn't imply u ∈ N(v).
        uknn      Σ_{v ∈ N(u), v read i} S(u, v)·(r_vi − b̂_vi) / (Σ S(u, v) + 0.5)
        uknn_den  log(1 + Σ_{v ∈ N(u), v read i} S(u, v))
    NaN where the user or item is unknown; 0 where no neighbour read the item.
    Similarities are computed in dense blocks: memory ≈ block × n_users × 4 bytes.
    """

    def __init__(self, top_n: int = 100, block: int = 2000, chunk: int = 20_000) -> None:
        self.top_n, self.block, self.chunk = top_n, block, chunk

    def _neighbours(self, R: csr_matrix) -> csr_matrix:
        """user × user matrix keeping each user's top_n similarities."""
        idf = np.log1p(
            R.shape[0] / np.maximum(np.asarray(R.sum(axis=0)).ravel(), 1))
        X = normalize(R.multiply(idf).tocsr())
        rows, cols, vals = [], [], []
        for s in range(0, X.shape[0], self.block):
            sims = (X[s:s + self.block] @ X.T).toarray().astype(np.float32)
            at = np.arange(sims.shape[0])
            sims[at, s + at] = 0
            top = np.argpartition(-sims, self.top_n, axis=1)[:, :self.top_n]
            v = sims[np.repeat(at, self.top_n), top.ravel()]
            keep = v > 0
            rows.append(np.repeat(s + at, self.top_n)[keep])
            cols.append(top.ravel()[keep])
            vals.append(v[keep])
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(R.shape[0], R.shape[0]))

    def fit(self, train: pd.DataFrame, base: RatingModel) -> Self:
        u, self.users = pd.factorize(train["user"])
        i, self.items = pd.factorize(train["item"])
        res = train["rating"].to_numpy(dtype=float) - base.predict(train)
        R = csr_matrix((np.ones(len(train)), (u, i)),
                       shape=(len(self.users), len(self.items)))
        self.neigh_ = self._neighbours(R)
        # item × user; the tiny offset keeps an exact-zero residual as a stored entry
        self.res_ = csr_matrix((res + 1e-9, (i, u)),
                               shape=(len(self.items), len(self.users)))
        self.seen_ = R.T.tocsr()
        return self

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        u = self.users.get_indexer(rows["user"])
        i = self.items.get_indexer(rows["item"])
        known = np.flatnonzero((u >= 0) & (i >= 0))
        out = np.full((len(rows), 2), np.nan)
        for s in range(0, len(known), self.chunk):
            where = known[s:s + self.chunk]
            N = self.neigh_[u[where]]
            num = np.asarray(N.multiply(
                self.res_[i[where]]).sum(axis=1)).ravel()
            den = np.asarray(N.multiply(
                self.seen_[i[where]]).sum(axis=1)).ravel()
            out[where, 0], out[where, 1] = num / (den + 0.5), np.log1p(den)
        return pd.DataFrame(out, columns=["uknn", "uknn_den"])
