from typing import Tuple
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dispersion(
    W: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:

    W = np.asarray(W)

    CV2 = np.var(W, axis=1) / (np.mean(W, axis=1) ** 2)
    CV2_nrm = 1 - CV2 / (len(W) - 1)

    return CV2, CV2_nrm


dispersion.__doc__ = files("abct").joinpath("docstrings", "dispersion.md").read_text()
