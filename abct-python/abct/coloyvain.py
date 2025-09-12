import warnings
from typing import Tuple
from importlib.resources import files

import numpy as np
from abct import loyv

def coloyvain(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    ## Parse, process, and test arguments

    # Parse arguments
    Args = loyv.step0_args("coloyvain", *args, **kwargs)

    # Process initial arguments
    Args = loyv.step1_proc_coloyvain(Args)

    # Test arguments
    loyv.step2_test(Args.X, Args.W, Args.px, Args.k, Args)
    loyv.step2_test(Args.Y, Args.W, Args.py, Args.k, Args)

    ## Run algorithm

    R = -np.inf
    for i in range(Args.replicates):
        Args.replicate_i = i

        # initialize
        Mx0 = loyv.step3_init([], [], Args.DistX, Args.px, Args)
        My0 = loyv.step3_init([], [], Args.DistY, Args.py, Args)

        # get between-module correlations
        MMx0 = np.zeros((Args.k, Args.px))
        MMx0[Mx0, np.arange(Args.px)] = 1
        MMy0 = np.zeros((Args.k, Args.py))
        MMy0[My0, np.arange(Args.py)] = 1

        match Args.objective:
            case "cokmeans":
                Ox = np.ones((Args.px, 1))
                Oy = np.ones((Args.py, 1))
            case "cospectral":
                Ox = np.sum(Args.W, axis=1, keepdims=True)
                Oy = np.sum(Args.W, axis=0, keepdims=True).T

        C0_nrm = (MMx0 @ Args.W @ MMy0.T) / np.sqrt((MMx0 @ Ox) @ (MMy0 @ Oy).T)

        # align modules
        Mx1 = np.zeros_like(Mx0)
        My1 = np.zeros_like(My0)
        for h in range(Args.k):
            ix, iy = np.unravel_index(np.nanargmax(C0_nrm), C0_nrm.shape)
            Mx1[Mx0 == ix] = h
            My1[My0 == iy] = h
            C0_nrm[ix, :] = np.nan
            C0_nrm[:, iy] = np.nan

        # fixed point iteration until convergence
        for v in range(Args.maxiter):
            My0 = My1.copy()
            Mx1,  _,      _ = loyv.step4_run(Args, Args.W, Mx1, My1)  # optimize Mx
            My1, R1, R1_all = loyv.step4_run(Args, Args.W.T, My1, Mx1)  # optimize My
            if np.array_equal(My0, My1):  # if identical, neither Mx1 nor My1 will change
                break
            if Args.display == "iteration":
                print(f"Replicate: {Args.replicate_i:4d}.    Iteration: {v:4d}.    Objective: {R1:4.4f}.")
            if v == Args.maxiter - 1:
                warnings.warn(f"Algorithm did not converge after {v + 1} iterations.")

        # check if replicate has improved on previous result
        if (R1 - R) > Args.tolerance:  # test for increase
            if Args.display in ["replicate", "iteration"]:
                print(f"Replicate: {i:4d}.    Objective: {R1:4.4f}.    Δ: {R1 - R:4.4f}.")
            R = R1
            Mx = Mx1
            My = My1
            R_all = R1_all

    return Mx, My, R, R_all.ravel()


coloyvain.__doc__ = files("abct").joinpath("docstrings", "doc_coloyvain.py").read_text().replace("```", "")
