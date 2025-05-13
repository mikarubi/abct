from numpy.typing import ArrayLike
from pydantic import validate_call
from .utils import get_docstring

import numpy as np


@validate_call
def coneighbors(W: ArrayLike, thr: float = 0.1) -> np.ndarray:

    W = np.asarray(W)

    # Create binary matrix of top neighbors
    A = W > np.quantile(W, 1 - thr, axis=1, keepdims=True)

    # Compute common neighbors matrix
    return A @ A.T


coneighbors.__doc__ = get_docstring.from_matlab("coneighbors.m")
