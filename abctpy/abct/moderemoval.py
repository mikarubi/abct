from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict

import numpy as np
from scipy import sparse


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def moderemoval(
    X: ArrayLike, type: Literal["degree", "global", "rankone", "soft"] = "degree"
) -> np.ndarray:

    X = np.asarray(X)

    match type:
        # Degree correction
        case "degree":
            if not np.all(X >= 0):
                raise ValueError("Invalid degree correction: Matrix must be non-negative.")
            So = np.sum(X, axis=1)
            Si = np.sum(X, axis=0)
            return X - np.outer(So, Si) / np.sum(So)

        # Global signal regression
        case "global":
            G = np.mean(X, axis=0, keepdims=True)
            return X - (X @ G.T) @ G / (G @ G.T)

        # Subtraction of rank-one approximation
        case "rankone":
            U, S, VT = sparse.linalg.svds(X, k=1)
            return X - U @ np.diag(S) @ VT

        # Soft removal of primary modes
        case "soft":
            if not (X.shape[0] == X.shape[1] and np.allclose(X, X.T)):
                raise ValueError("Invalid soft mode removal: Network matrix must be symmetric.")

            # Compute eigendecomposition
            D, V = np.linalg.eigh(X)
            idx = np.argsort(D)[::-1]
            V = V[:, idx]
            D = D[idx]

            n = len(D)

            # Fit cubic polynomial to all eigenvalues
            bk, r0 = np.polyfit(np.arange(1, n + 1), D, 3, full=True)[:2]
            rms0 = np.sqrt(r0[0] / n)  # get rms

            # Rescale x to [0, 1]
            x = (np.arange(1, n + 1) - 1) / (n - 1)
            y = np.zeros(n)

            # Find optimal fit
            for k in range(n):
                b = bk
                bk, rk = np.polyfit(np.arange(k + 1, n + 1), D[k:], 3, full=True)[:2]
                assert len(np.arange(k + 1, n + 1)) == (n - k)
                rmsk = np.sqrt(rk[0] / (n - k))
                y[k] = (rms0 - rmsk) / rms0
                # Detect knee of optimal fit and break
                if k > 0 and (y[k] - x[k]) < (y[k - 1] - x[k - 1]):
                    break

            # Apply the optimal fit
            D = np.polyval(b, np.arange(1, n + 1))
            return V @ np.diag(D) @ V.T


with open("abct/docs/moderemoval", "r") as f:
    moderemoval.__doc__ = f.read()
