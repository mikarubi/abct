from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import abct
import numpy as np

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def degrees(
    W: ArrayLike,
    type: Literal["first", "second", "residual"] = "first"
) -> np.ndarray:

    W = np.asarray(W)
    match type:
        case "first":
            return np.sum(W, axis=1)
        case "second":
            return np.sum(W**2, axis=1)
        case "residual":
            W_residual = abct.residualn(W, "rankone")
            return np.sum(W_residual, axis=1)


degrees.__doc__ = resources.read_text("abct.docstrings", "degrees")
