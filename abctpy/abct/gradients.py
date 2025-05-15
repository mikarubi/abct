import warnings
from typing import Literal, Optional
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import abct
import numpy as np
from scipy import sparse

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
        B = abct.coneighbors(W)
    else:
        B = abct.coneighbors(W, thr)

    # Get gradients
    match weight:
        case "weighted":
            if kwargs:
                warnings.warn("Ignoring Name=Value arguments for weighted gradients.")
            _, V = sparse.linalg.eigs(B, k=k + 1)
            return V[:, 1:]  # Remove first eigenvector
        case "binary":
            M = abct.loyvain(B, k, "kmodularity", "network", **kwargs)
            V = np.zeros((len(M), k))
            V[np.arange(len(M)), M] = 1
            return V


gradients.__doc__ = resources.read_text("abct.docs", "gradients")
