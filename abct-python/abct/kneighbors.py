from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import numpy as np
import pynndescent
from scipy import sparse

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def kneighbors(
    X: ArrayLike,
    type: Literal["common", "nearest"] = "common",
    k: float = 10,
    similarity: Literal["network", "corr", "cosim"] = "network",
    method: Literal["direct", "indirect"] = "direct",
    **kwargs,
) -> sparse.csr_matrix:

    n = X.shape[0]

    if k < 1:
        k = np.clip(np.round(n * k), 1, n-1)
    else:
        assert k < n, "k must be less than number of nodes or data points."
        assert np.isclose(k, np.round(k)), "k > 1 must be an integer."
    k = int(k)

    Row = np.tile(np.r_[:n][:, None], (1, k+1))
    if similarity == "network":
        W = X.copy()
        assert(W.shape[0] == W.shape[1] and np.allclose(W, W.T),
            "Network matrix must be symmetric or similarity must not be ""network"".")
        np.fill_diagonal(W, np.inf)
        Col = np.argpartition(W, -(k+1), axis=1)[:, -(k+1):].ravel()
    else:
        match method:
            case "direct":
                # Center to mean 0 for correlation
                if similarity == "corr":
                    X = X - np.mean(X, axis=1, keepdims=True)
                # Rescale to norm 1
                X = X / np.linalg.norm(X, axis=1, keepdims=True)

                # Compute similarity matrix in blocks of 1e8 elements
                # It follows that n * nb = 1e8, b = n / nb = n^2 / 1e8
                b = int(np.ceil(n**2 / 1e8))
                b = np.clip(b, 1, n)
                Ix = np.floor(np.linspace(0, n, b+1)).astype(int)
                Col = np.zeros((n, k+1), dtype=int)
                for i in range(b):
                    Ixi = np.r_[Ix[i]:Ix[i+1]]
                    Col[Ixi] = np.argpartition(X[Ixi, :] @ X.T, -(k+1), axis=1)[:, -(k+1):]

            case "indirect":
                match similarity:
                    case "corr": knnsim = "correlation"
                    case "cosim": knnsim = "cosine"
                knn_search_index = pynndescent.NNDescent(X, n_neighbors=k+1, metric=knnsim, **kwargs)
                Col, _ = knn_search_index.neighbor_graph

    A = sparse.csr_matrix((np.ones(n*(k+1)), (Row.ravel(), Col.ravel())), shape=(n, n))
    A.setdiag(0)

    match type:
        case "common":
            B = A @ A.T
        case "nearest":
            B = A.maximum(A.T)

    return B

kneighbors.__doc__ = resources.read_text("abct.docs", "kneighbors")
