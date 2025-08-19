import warnings
from typing import Literal, Tuple
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import abct
import numpy as np
from scipy import linalg, sparse

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def canoncov(
    X: ArrayLike,
    Y: ArrayLike,
    k: int,
    type: Literal["weighted", "binary"] = "weighted",
    corr: bool = False,
    resid: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Basic checks
    n, p = X.shape
    n_, q = Y.shape
    if n != n_:
        raise ValueError("X and Y must have the same number of data points.")
    if k > min(p, q):
        raise ValueError("k must not exceed number of features in X or Y.")

    # Initial processing
    if type == "weighted":
        if kwargs:
            warnings.warn("Ignoring Name=Value arguments for weighted analysis.")

    # Global residualization or centering
    if resid:  # Degree correction automatically centers data
        X = abct.residualn(X, "degree")
        Y = abct.residualn(Y, "degree")
    else:
        X = X - np.mean(X, axis=0, keepdims=True)
        Y = Y - np.mean(Y, axis=0, keepdims=True)

    # Set up problem
    if corr:
        Ux, Sx, VxT = linalg.svd(X, full_matrices=False)
        Uy, Sy, VyT = linalg.svd(Y, full_matrices=False)
        Z = VxT.T @ Ux.T @ Uy @ VyT
    else:
        Z = X.T @ Y

    # Solve problem
    if type == "weighted":
        A, R, B = sparse.linalg.svds(Z, k=k)
        R = np.diag(R)
        B = B.T
    else:
        Mx, My, _, R = abct.coloyvain(Z, k, "kmeans", "network", numbatches=min(32, min(p, q)), **kwargs)
        ix = np.argsort(R)[::-1]
        R = R[ix]
        A = np.zeros((p, k))
        B = np.zeros((q, k))
        for h in range(k):
            A[Mx == ix[h], h] = 1
            B[My == ix[h], h] = 1

    # Recover coefficients
    if corr:
        A = np.linalg.lstsq(VxT, Sx.T)[0].T @ VxT @ A
        B = np.linalg.lstsq(VyT, Sy.T)[0].T @ VyT @ B

    U = X @ A
    V = Y @ B

    return A, B, U, V, R


canoncov.__doc__ = resources.read_text("abct.docstrings", "canoncov")
