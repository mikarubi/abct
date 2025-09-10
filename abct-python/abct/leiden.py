from typing import Literal
from pydantic import validate_call, ConfigDict
from importlib.resources import files

import numpy as np
import igraph as ig

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def leiden(
    W,
    start,
    gamma: float = 1,
    replicates: int = 10,
    finaltune: bool = False,
    tolerance: float = 1e-10,
    display: Literal["none", "replicate"] = "none",
):
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
        )
        if finaltune:
            C = G.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=gamma,
                initial_membership=C.membership,
            )

        # test for increase
        if (C.modularity - Q) > tolerance:
            if display == "replicate":
                print("Replicate: %4d.    Objective: %4.4f.    Δ: %4.4f." % (i, C.modularity, C.modularity - Q))
            
            Q = C.modularity
            M = C.membership

    return M, Q

leiden.__doc__ = files("abct").joinpath("docstrings", "louvains.md")
