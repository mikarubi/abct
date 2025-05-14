from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources
import numpy as np


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def coneighbors(W: ArrayLike, thr: float = 0.1) -> np.ndarray:

    W = np.asarray(W)

    # Create binary matrix of top neighbors
    A = W > np.quantile(W, 1 - thr, axis=1, keepdims=True)

    # Compute common neighbors matrix
    return A @ A.T


coneighbors.__doc__ = resources.read_text("abct.docs", "coneighbors")
