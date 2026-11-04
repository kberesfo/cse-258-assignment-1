from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


@dataclass
class CosineFit:
    """Item-item cosine similarity over user sets, fitted on one training set."""

    users: pd.Index     # row order of R
    items: pd.Index     # column order of R; row/column order of S
    R: csr_matrix       # user × item, 0/1 history
    S: csr_matrix       # item × item cosine, diagonal zeroed


def fit_item_cosine(train_df: pd.DataFrame) -> CosineFit:
    """cos(i, j) = |U_i ∩ U_j| / sqrt(|U_i| |U_j|), computed as normalized item vectors' dot products."""
    users = pd.Index(train_df["user"].unique())
    items = pd.Index(train_df["item"].unique())

    R = csr_matrix(
        (
            np.ones(len(train_df)),
            (
                users.get_indexer(train_df["user"]),
                items.get_indexer(train_df["item"])
            )
        ),
        shape=(len(users), len(items)),
    )

    R.data[:] = 1                         # repeat interactions count once
    Xn = normalize(R.T.tocsr())           # item × user, each row unit length
    S = (Xn @ Xn.T).tocsr()               # item × item cosine
    S.setdiag(0)                          # an item isn't evidence for itself
    S.eliminate_zeros()

    return CosineFit(users=users, items=items, R=R, S=S)


def cos_features(fit: CosineFit, users: pd.Series, items: pd.Series) -> pd.DataFrame:
    """
    Similarity of each candidate to the user's training history, using item-item cosine.

    cos_sum:  Σ_{j ∈ H_u} cos(i, j)    — total support from the history
    cos_max:  max_{j ∈ H_u} cos(i, j)  — closest single history item
    cos_mean: cos_sum / |H_u|          — support normalized by history length
    Unknown users or items get 0 for all three.
    """
    u = fit.users.get_indexer(users)
    i = fit.items.get_indexer(items)

    ok = (u >= 0) & (i >= 0)
    out = {k: np.zeros(len(u)) for k in ("cos_sum", "cos_max", "cos_mean")}

    H = fit.R[u[ok]]                          # each row: that user's history
    # similarities to history items only
    P = H.multiply(fit.S[i[ok]]).tocsr()
    s = np.asarray(P.sum(axis=1)).ravel()
    n = np.asarray(H.sum(axis=1)).ravel()
    
    out["cos_sum"][ok] = s
    out["cos_max"][ok] = P.max(axis=1).toarray().ravel()
    out["cos_mean"][ok] = s / np.maximum(n, 1)
    
    return pd.DataFrame(out)
