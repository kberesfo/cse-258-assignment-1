"""Item-item similarity over user interactions (PML §4.3).

Shared by the ranking features (sum / max / mean similarity to a user's history, and
LocalSimilarity over the reads nearest a query time) and the rating kNN (similarity-weighted
residuals). A similarity is a function (R, Rres) -> S:
    R:    user × item 0/1 history
    Rres: user × item residuals r − b̂ (only needed by rating-based measures)
    S:    item × item similarity, diagonal zeroed; S[candidate, history item]. The measures here
          are symmetric; ranking/ease.py's ease_similarity is not.
"""

from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

Similarity = Callable[[csr_matrix, csr_matrix | None], csr_matrix]


def _no_diag(S) -> csr_matrix:
    S = S.tocsr()
    S.setdiag(0)              # an item isn't evidence for itself
    S.eliminate_zeros()
    return S


def _shrink_by_cosupport(S: csr_matrix, R: csr_matrix, shrink: float) -> csr_matrix:
    """Multiply each S_ij by n_ij / (n_ij + shrink), n_ij = users who interacted with both."""
    C = (R.T @ R).tocsr()
    C.data = C.data / (C.data + shrink)
    return S.multiply(C).tocsr()


def binary_cosine(R: csr_matrix, Rres: csr_matrix | None = None) -> csr_matrix:
    """Co-reading: |U_i ∩ U_j| / sqrt(|U_i| |U_j|)."""
    X = normalize(R.T.tocsr())
    return _no_diag(X @ X.T)


def jaccard(R: csr_matrix, Rres: csr_matrix | None = None) -> csr_matrix:
    """Co-reading: |U_i ∩ U_j| / |U_i ∪ U_j|."""
    C = (R.T @ R).tocoo()
    n = np.asarray(R.sum(axis=0)).ravel()
    C.data = C.data / (n[C.row] + n[C.col] - C.data)
    return _no_diag(C)


def adjusted_cosine(R: csr_matrix, Rres: csr_matrix | None, shrink: float = 10.0) -> csr_matrix:
    """Agreement: do users who beat expectations on i also beat them on j? (can be < 0)."""
    if Rres is None:
        raise ValueError("adjusted_cosine needs residuals")
    X = normalize(Rres.T.tocsr())
    S = X @ X.T
    return _no_diag(_shrink_by_cosupport(S, R, shrink) if shrink else S)


class ItemSimilarity:
    """Item-item similarity fitted on one training set, plus features over a user's history."""

    def __init__(self, similarity: Similarity = binary_cosine, prefix: str = "cos", top_k: tuple[int, ...] = ()) -> None:
        self.similarity = similarity
        self.prefix = prefix
        self.top_k = top_k
        self.S: csr_matrix | None = None

    def fit(self, train: pd.DataFrame, residuals: NDArray | None = None) -> Self:
        """residuals: r − b̂ aligned with train's rows (needed only by rating-based similarities)."""
        u, self.users = pd.factorize(train["user"])
        i, self.items = pd.factorize(train["item"])
        shape = (len(self.users), len(self.items))
        self.R = csr_matrix((np.ones(len(train)), (u, i)), shape=shape)
        # repeat interactions count once
        self.R.data[:] = 1
        self.Rres = None if residuals is None else csr_matrix(
            (residuals, (u, i)), shape=shape)
        self.S = self.similarity(self.R, self.Rres)
        return self

    def features(self, users: pd.Series, items: pd.Series) -> pd.DataFrame:
        """{prefix}_sum / _max / _mean similarity of each candidate to the user's history (0 if unknown)."""
        if self.S is None:
            raise RuntimeError("ItemSimilarity.features called before fit")
        u = self.users.get_indexer(users)
        i = self.items.get_indexer(items)
        ok = (u >= 0) & (i >= 0)
        p = self.prefix
        out = {f"{p}_{k}": np.zeros(len(u)) for k in ("sum", "max", "mean")}

        # each row: that user's history
        H = self.R[u[ok]]
        # similarities to history items only
        P = H.multiply(self.S[i[ok]]).tocsr()
        s = np.asarray(P.sum(axis=1)).ravel()
        n = np.asarray(H.sum(axis=1)).ravel()

        out[f"{p}_sum"][ok] = s
        out[f"{p}_max"][ok] = P.max(axis=1).toarray().ravel()
        out[f"{p}_mean"][ok] = s / np.maximum(n, 1)

        if self.top_k:
            C = P.tocoo()                                       # similarities to history items
            # by row, largest first
            order = np.lexsort((-C.data, C.row))
            r, d = C.row[order], C.data[order]
            # first entry of each row
            starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
            rank = np.arange(len(r)) - np.repeat(starts,
                                                 np.diff(np.r_[starts, len(r)]))
            n_rows = P.shape[0]
            for k in self.top_k:
                keep = rank < k
                topk = np.bincount(
                    r[keep], weights=d[keep], minlength=n_rows) / k
                out[f"{p}_top{k}"] = np.zeros(len(u))
                out[f"{p}_top{k}"][ok] = topk

        return pd.DataFrame(out)
