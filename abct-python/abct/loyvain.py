from typing import Tuple
from importlib.resources import files

import numpy as np
from . import loyv

def loyvain(*args, **kwargs) -> Tuple[np.ndarray, float]:

    ## Parse, process, and test arguments

    # Parse arguments
    Args = loyv.step0_args("loyvain", *args, **kwargs)

    # Process all other arguments
    Args = loyv.step1_proc_loyvain(Args)

    # Test arguments
    loyv.step2_test(Args.X, Args.W, Args.n, Args.k, Args)

    ## Run algorithm

    Q = -np.inf
    for i in range(Args.replicates):
        Args.replicate_i = i
        if Args.start == "custom":
            M0 = Args.M0
        else:
            # initialize
            M0 = loyv.step3_init(Args.X, Args.normX, Args.Dist, Args.n, Args)

        # run algorithm
        M1, Q1, _ = loyv.step4_run(Args, Args.W, M0)

        # test for increase
        if (Q1 - Q) > Args.tolerance:
            if Args.display in ["replicate", "iteration"]:
                print(
                    f"Replicate: {i:4d}.    Objective: {Q1:4.4f}.    Δ: {Q1 - Q:4.4f}."
                )
            Q = Q1
            M = M1

    return M, Q


loyvain.__doc__ = files("abct").joinpath("docstrings", "doc_loyvain.py").read_text().replace("```", "")
