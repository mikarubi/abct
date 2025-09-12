from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def shrinkage(X: ArrayLike) -> np.ndarray:

    X = np.asarray(X)

    if not ((X.shape[0] == X.shape[1]) and np.allclose(X, X.T)):
        raise ValueError("Network matrix must be symmetric.")

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
        if k > 0 and (y[k] - x[k]) < (y[k-1] - x[k-1]):
            break

    # Apply the optimal fit
    D = np.polyval(b, np.arange(1, n + 1))
    return V @ np.diag(D) @ V.T


shrinkage.__doc__ = files("abct").joinpath("docstrings", "doc_shrinkage.py").read_text().replace("```", "")
