import warnings
from typing import Literal, Optional
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import abct
import numpy as np
from scipy import sparse

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def conicomps(
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

    # Get components
    match weight:
        case "weighted":
            if kwargs:
                warnings.warn("Ignoring Name=Value arguments for weighted components.")
            _, V = sparse.linalg.eigs(B, k=k + 1)
            assert not np.any(np.imag(V)), "Weighted components should be real-valued"
            V = np.real(V)
            return V[:, 1:]  # Remove first eigenvector
        case "binary":
            M = abct.loyvain(B, k, "kmodularity", "network", **kwargs)[0]
            V = np.zeros((len(M), k))
            V[np.arange(len(M)), M] = 1
            return V


conicomps.__doc__ = resources.read_text("abct.docs", "conicomps")
