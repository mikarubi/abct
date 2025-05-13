from typing import Literal, ArrayLike
from pydantic import validate_call

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
            Y = 0
        case "coloyvain":
            n_args_num = 2 + (n_args > 2 and isinstance(args[2], (int, float)))
            if n_args_num == 2:
                W, k = args[:n_args_num]
                X, Y = 0, 0
            elif n_args_num == 3:
                X, Y, k = args[:n_args_num]
                W = 0
    args = args[n_args_num:]
    if n_args >= n_args_num + 1:
        kwargs["objective"] = args[0]
        args = args[1:]
        if n_args >= n_args_num + 2:
            kwargs["similarity"] = args[0]
            args = args[1:]

    return parse_args(method=method, W=W, X=X, Y=Y, k=k, **kwargs)


@validate_call
def parse_args(
    method: Literal["loyvain", "coloyvain"],
    W: ArrayLike = 0,
    X: ArrayLike = 0,
    Y: ArrayLike = 0,
    k: int = 0,
    objective: Literal["kmodularity", "kmeans", "spectral"] = "kmodularity",
    similarity: Literal["network", "corr", "cosim", "cov", "dot"] = "network",
    start: Literal["greedy", "balanced", "random"] = "greedy",
    numbatches: int = 10,
    maxiter: int = 1000,
    replicates: int = 10,
    tolerance: float = 1e-10,
    display: Literal["none", "replicate", "iteration"] = "none",
) -> dict:

    if isinstance(start, (int, float)) and method == "loyvain":
        pass
    elif isinstance(start, str) and start in ["greedy", "balanced", "random"]:
        pass
    else:
        raise ValueError('Start must be either "greedy", "balanced", "random", or a numeric vector (loyvain only).')

    if method == "coloyvain":
        if k <= 0:
            raise ValueError("k must be positive for co-Loyvain.")
        if similarity == "network":
            if not (np.array_equal(X, 0) and np.array_equal(Y, 0)):
                raise ValueError("X and Y inputs are incompatible with 'network' similarity.")
        else:
            if not np.array_equal(W, 0):
                raise ValueError("W input is only compatible with 'network' similarity.")
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y must have the same number of data points.")

    return {key: value for key, value in locals().items()}
