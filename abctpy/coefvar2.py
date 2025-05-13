from typing import Tuple
from numpy.typing import ArrayLike
from pydantic import validate_call
from .utils import get_docstring

import numpy as np


@validate_call
def coefvar2(
    W: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:

    W = np.asarray(W)

    CV2 = np.var(W, axis=1) / (np.mean(W, axis=1) ** 2)
    CV2_nrm = 1 - CV2 / (len(W) - 1)

    return CV2, CV2_nrm


coefvar2.__doc__ = get_docstring.from_matlab("coefvar2.m")
