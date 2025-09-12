from typing import Literal
from numpy.typing import ArrayLike
from scipy.sparse import sparray
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np
import pynndescent
from scipy import sparse

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def kneighbor(
    X: ArrayLike | sparray,
    type: Literal["common", "nearest"] = "common",
    kappa: float = 10,
    similarity: Literal["network", "corr", "cosim"] = "network",
    method: Literal["direct", "indirect"] = "direct",
    **kwargs,
) -> sparse.csr_array:

    n = len(X)

    if kappa < 1:
        kappa = np.clip(np.round(n * kappa), 1, n-1)
    else:
        assert kappa < n, "kappa must be less than number of nodes or data points."
        assert np.isclose(kappa, np.round(kappa)), "kappa > 1 must be an integer."
    kappa = int(kappa)

    Row = np.tile(np.arange(n)[:, np.newaxis], (1, kappa+1))
    if similarity == "network":
        W = X.copy()
        assert W.shape[0] == W.shape[1], "Network matrix must be square."
        np.fill_diagonal(W, np.inf)
        Col = np.argpartition(W, -(kappa+1), axis=1)[:, -(kappa+1):].ravel()
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
                Col = np.zeros((n, kappa+1), dtype=int)
                for i in range(b):
                    Ixi = np.arange(Ix[i], Ix[i+1])
                    Col[Ixi] = np.argpartition(X[Ixi, :] @ X.T, -(kappa+1), axis=1)[:, -(kappa+1):]

            case "indirect":
                match similarity:
                    case "corr": knnsim = "correlation"
                    case "cosim": knnsim = "cosine"
                knn_search_index = pynndescent.NNDescent(X, n_neighbors=kappa+1, metric=knnsim, **kwargs)
                Col, _ = knn_search_index.neighbor_graph

    A = sparse.csr_array((np.ones(n*(kappa+1)), (Row.ravel(), Col.ravel())), shape=(n, n))
    A.setdiag(0)

    match type:
        case "common":
            B = A @ A.T
        case "nearest":
            B = A.maximum(A.T)

    return B

kneighbor.__doc__ = files("abct").joinpath("docstrings", "doc_kneighbor.py").read_text().replace("```", "")
