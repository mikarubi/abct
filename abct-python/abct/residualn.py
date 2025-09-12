from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np
from scipy import sparse


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def residualn(
    X: ArrayLike,
    type: Literal["degree", "global", "rankone"] = "degree"
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


residualn.__doc__ = files("abct").joinpath("docstrings", "doc_residualn.py").read_text().replace("```", "")
