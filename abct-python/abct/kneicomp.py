import warnings
from typing import Literal, Optional
from numpy.typing import ArrayLike
from scipy.sparse import sparray
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import abct
import numpy as np
from scipy import sparse

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def kneicomp(
    W: ArrayLike | sparray,
    k: int,
    weight: Literal["weighted", "binary"] = "weighted",
    **kwargs,
) -> np.ndarray:

    W = np.asarray(W)

    # Default kneighbor arguments
    args = {
        "type": "common",
        "kappa": 0.1,
        "similarity": "network",
        "method": "direct",
    }
    # Update arguments with kwargs
    args.update(kwargs)
    kwargs = {key: kwargs[key] for key in set(kwargs.keys()) - set(args.keys())}

    # Get neighbors matrix
    B = abct.kneighbor(W, args["type"], args["kappa"], args["similarity"], args["method"])

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
            M = abct.loyvain(B.toarray(), k, "kmodularity", "network", **kwargs)[0]
            V = np.zeros((len(M), k))
            V[np.arange(len(M)), M] = 1
            return V


kneicomp.__doc__ = files("abct").joinpath("docstrings", "doc_kneicomp.py").read_text().replace("```", "")
