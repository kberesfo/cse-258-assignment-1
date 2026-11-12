"""EASE (Steck 2019), embarrassingly shallow autoencoder: a closed-form item-item model.

    B = argmin ||X − XB||² + λ||B||²  s.t. diag(B) = 0   →   B_ij = −P_ij / P_jj,  P = (XᵀX + λI)⁻¹

score(u, i) = Σ_{j ∈ history(u)} B_ji. B is dense (items × items): ~17k items (Task 4) is
~1.2 GB in float32 and a ~1-2 minute inversion; ~51k items (Task 2) needs dtype=float32, which
inverts in place (~10-11 GB peak) and takes several minutes.
"""

from typing import Self

import numpy as np
import pandas as pd
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..similarity import Similarity, _no_diag


def ease_weights(R: csr_matrix, lam: float, dtype=np.float64) -> NDArray[np.float32]:
    """B (items × items, diagonal 0) from a binary user × item matrix R.
    dtype=float32 builds and inverts the Gram matrix in place, at half the memory."""
    G = (R.T @ R).astype(dtype).toarray()
    G[np.diag_indices_from(G)] += lam
    if dtype == np.float64:
        P = np.linalg.inv(G)    # unchanged path: Task 4's shipped predictions
        del G
    else:
        P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
    d = np.diag(P).copy()
    P /= -d                     # column j divided by −P_jj
    np.fill_diagonal(P, 0.0)
    return P.astype(np.float32, copy=False)


class EASE:
    """EASE fitted on every (user, item) row; score(users, items) gives one exact score per row."""

    def __init__(self, lam: float = 500.0, chunk: int = 2000) -> None:
        self.lam, self.chunk = lam, chunk

    def fit(self, train: pd.DataFrame) -> Self:
        u, self.users = pd.factorize(train["user"])
        i, self.items = pd.factorize(train["item"])
        self.X = csr_matrix((np.ones(len(train), dtype=np.float32), (u, i)),
                            shape=(len(self.users), len(self.items)))
        self.X.data[:] = 1          # repeat interactions count once
        self.B = ease_weights(self.X, self.lam)
        return self

    def score(self, users: ArrayLike, items: ArrayLike) -> NDArray[np.float64]:
        """Σ_{j ∈ history(u)} B_ji per row; 0 where the user or item is unknown."""
        u = self.users.get_indexer(pd.Series(users))
        i = self.items.get_indexer(pd.Series(items))
        ok = np.flatnonzero((u >= 0) & (i >= 0))
        out = np.zeros(len(u))
        for a in range(0, len(ok), self.chunk):     # chunks: avoid a rows × items dense block
            r = ok[a:a + self.chunk]
            out[r] = np.asarray(self.X[u[r]].multiply(self.B[:, i[r]].T).sum(axis=1)).ravel()
        return out

    __call__ = score


def ease_similarity(lam: float = 500.0, keep: int = 200, dtype=np.float64) -> Similarity:
    """EASE weights as a Similarity for ItemSimilarity / LocalSimilarity: S = Bᵀ (row = candidate,
    column = history item), sparsified to the `keep` largest |weights| per candidate.
    ItemSimilarity's {prefix}_sum is then ≈ the EASE score. Unlike cosine, S is asymmetric and
    can be negative. dtype: see ease_weights (float32 for Task 2's 51k items)."""

    def similarity(R: csr_matrix, Rres: csr_matrix | None = None) -> csr_matrix:
        S = ease_weights(R, lam, dtype).T           # S[c, h] = B[h, c]
        n = S.shape[0]
        k = min(keep, n - 1)
        rows, cols, vals = [], [], []
        for a in range(0, n, 1000):
            blk = np.ascontiguousarray(S[a:a + 1000])
            idx = np.argpartition(-np.abs(blk), k, axis=1)[:, :k]
            rows.append(np.repeat(np.arange(a, a + len(blk)), k))
            cols.append(idx.ravel())
            vals.append(np.take_along_axis(blk, idx, 1).ravel())
        return _no_diag(csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                           np.concatenate(cols))), shape=(n, n)))

    return similarity
