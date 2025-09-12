from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import abct
import numpy as np

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def degree(
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


degree.__doc__ = files("abct").joinpath("docstrings", "doc_degree.py").read_text().replace("```", "")
