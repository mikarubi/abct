import warnings
from typing import Literal, Optional
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict

import numpy as np
from scipy import sparse
from .coneighbors import coneighbors
from .loyvain import loyvain


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def gradients(
    W: ArrayLike,
    k: int,
    weight: Literal["weighted", "binary"] = "weighted",
    thr: Optional[float] = None,
    **kwargs,
) -> np.ndarray:

    W = np.asarray(W)

    # Get common-neighbors matrix
    if thr is None:
        B = coneighbors(W)
    else:
        B = coneighbors(W, thr)

    # Get gradients
    match weight:
        case "weighted":
            if kwargs:
                warnings.warn("Ignoring Name=Value arguments for weighted gradients.")
            _, V = sparse.linalg.eigs(B, k=k + 1)
            return V[:, 1:]  # Remove first eigenvector
        case "binary":
            M = loyvain(B, k, "kmodularity", "network", **kwargs)
            V = np.zeros((len(M), k))
            V[np.arange(len(M)), M] = 1
            return V


with open("abctpy/docs/gradients", "r") as f:
    gradients.__doc__ = f.read()
