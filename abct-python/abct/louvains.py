from typing import Literal
from numpy.typing import ArrayLike
from pydantic import validate_call, ConfigDict
from importlib import resources

import numpy as np
import igraph as ig

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def louvains(
    W: ArrayLike,
    gamma: float = 1,
    start: ArrayLike = None,
    replicates: int = 10,
    finaltune: bool = False,
    tolerance: float = 1e-10,
    display: Literal["none", "replicate"] = "none",
):
    if np.any(W < 0) or not ((W.shape[0] == W.shape[1]) and np.allclose(W, W.T)):
        raise ValueError("Matrix must be nonnegative and symmetric.")

    G = ig.Graph.Weighted_Adjacency(W, mode="undirected", attr="weight")
    Q = -np.inf
    for i in range(replicates):
        replicate_i = i

        # run algorithm
        C = G.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=gamma,
            initial_membership=start,
            n_iterations=-1,
        )
        if finaltune:
            C = G.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=gamma,
                initial_membership=C.membership,
                n_iterations=-1,
            )

        # test for increase
        if (C.modularity - Q) > tolerance:
            if display == "replicate":
                print("Replicate: %4d.    Objective: %4.4f.    Δ: %4.4f." % (i, C.modularity, C.modularity - Q))
            
            Q = C.modularity
            M = C.membership

    return M, Q

louvains.__doc__ = resources.read_text("abct.docstrings", "louvains")
