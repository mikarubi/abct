from typing import Literal
from numpy.typing import ArrayLike
from scipy.sparse import sparray as SparseArray
from pydantic import validate_call, ConfigDict
from types import SimpleNamespace

import torch
import numpy as np

def step0_args(X: ArrayLike, **kwargs) -> SimpleNamespace:
    # m-umap arguments initialization

    Args = parse_args(X=X, **kwargs)

    return SimpleNamespace(**Args)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def parse_args(
    X: ArrayLike | SparseArray,
    d: int = 3,
    kappa: float = 30,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    similarity: Literal["network", "corr", "cosim"] = "network",
    method: Literal["direct", "indirect"] = "direct",
    replicates: int = 10,
    finaltune: bool = True,
    partition: ArrayLike = None,
    start: ArrayLike | Literal["greedy", "spectral", "spectral_nn"] = "greedy",
    solver: Literal["adam", "trustregions"] = "adam",
    maxiter: int = 10000,
    learnrate: float = 1e-3,
    tolerance: float = 1e-6,
    gpu: bool = False,
    cache: bool = False,
    verbose: bool = True,
) -> dict:

    X = np.asarray(X)
    if similarity == "network":
        if (X.shape[0] != X.shape[1]) or not np.allclose(X, X.T):
            raise ValueError('Network matrix must be symmetric or similarity must not be "network".')

    if gpu and not torch.cuda.is_available():
        raise ValueError("GPU must be available or gpu must be False.")

    return {key: value for key, value in locals().items()}
