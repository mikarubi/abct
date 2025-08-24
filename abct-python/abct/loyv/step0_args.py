from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from types import SimpleNamespace

import numpy as np

def step0_args(method: str, *args, **kwargs) -> dict:
    # Loyvain arguments initialization

    n_args = len(args)
    if n_args < 2:
        raise ValueError("Wrong number of input arguments.")
    match method:
        case "loyvain":
            n_args_num = 2
            W, k = args[:n_args_num]
            X = W
            Y = np.array(None)
        case "coloyvain":
            n_args_num = 2 + (n_args > 2 and (not isinstance(args[2], str)))
            if n_args_num == 2:
                W, k = args[:n_args_num]
                X, Y = np.array(None), np.array(None)
            elif n_args_num == 3:
                X, Y, k = args[:n_args_num]
                W = np.array(None)
    args = args[n_args_num:]
    if n_args >= n_args_num + 1:
        kwargs["objective"] = args[0]
        args = args[1:]
        if n_args >= n_args_num + 2:
            kwargs["similarity"] = args[0]
            args = args[1:]

    Args = parse_args(method=method, W=W, X=X, Y=Y, k=k, **kwargs)

    return SimpleNamespace(**Args)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def parse_args(
    method: Literal["loyvain", "coloyvain"],
    W: ArrayLike = None,
    X: ArrayLike = None,
    Y: ArrayLike = None,
    k: int = 0,
    objective: Literal["kmodularity", "kmeans", "spectral"] = "kmodularity",
    similarity: Literal["network", "corr", "cosim", "cov", "dot"] = "network",
    start: ArrayLike | Literal["greedy", "balanced", "random"] = "greedy",
    numbatches: int = 10,
    maxiter: int = 1000,
    replicates: int = 10,
    tolerance: float = 1e-10,
    display: Literal["none", "replicate", "iteration"] = "none",
) -> dict:
    
    W = np.asarray(W)
    X = np.asarray(X)
    Y = np.asarray(Y)

    if method == "coloyvain":
        if isinstance(start, np.ndarray):
            raise ValueError("Start cannot be a numeric vector for co-Loyvain.")
        if k <= 0:
            raise ValueError("k must be positive for co-Loyvain.")
        if similarity == "network":
            if not (np.array_equal(X, None) and np.array_equal(Y, None)):
                raise ValueError('X and Y inputs are incompatible with "network" similarity.')
        else:
            if not np.array_equal(W, None):
                raise ValueError('W input is only compatible with "network" similarity.')
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y must have the same number of data points.")

    return {key: value for key, value in locals().items()}
