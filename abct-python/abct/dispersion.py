from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dispersion(
    W: ArrayLike,
    type: Literal["coefvar2", "kpartcoef"] = "coefvar2",
    M: ArrayLike = [],
) -> np.ndarray:

    W = np.asarray(W)
    M = np.asarray(M)

    match type:
        case "coefvar2":
            D = np.var(W, axis=1) / (np.mean(W, axis=1) ** 2)
        case "kpartcoef":
            # Basic checks
            n, n_ = W.shape
            n__ = M.size
            if n != n_ or n != n__:
                raise ValueError("W must be a square matrix and M must have the same length as W.")

            k = np.max(M) + 1
            MM = np.zeros((n, k))
            MM[np.arange(n), M] = 1
            kSnm = (W @ MM) / MM.sum(0, keepdims=True)
            P = kSnm / kSnm.sum(1, keepdims=True)
            D = 1 - (P**2).sum(1)

    return D


dispersion.__doc__ = files("abct").joinpath("docstrings", "doc_dispersion.py").read_text().replace("```", "")
