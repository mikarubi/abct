from types import SimpleNamespace

import abct
import numpy as np

def step1_proc_loyvain(Args):
    # Loyvain arguments processing

    Args = SimpleNamespace(**Args)

    Args.n, Args.p = Args.X.shape
    if Args.similarity == "network":
        Args.W = Args.X
        Args.X = 0
    else:
        Args.W = 0

    # Process custom initial module assignment
    if isinstance(Args.start, (int, float, np.ndarray)):
        Args.M0 = Args.start
        if Args.k == 0:
            Args.k = np.max(Args.M0)
        Args.start = "custom"

    # Remove first mode for kmodularity
    if Args.objective == "kmodularity":
        if Args.similarity == "network":
            Args.W = Args.W * (Args.n / Args.k) / np.sum(np.abs(Args.W))
            Args.W = abct.moderemoval(Args.W, "degree")
        else:
            Args.X = abct.moderemoval(Args.X, "global")
        Args.objective = "kmeans"

    # Center to mean 0 for covariance and correlation
    if Args.similarity in ["cov", "corr"]:
        Args.X = Args.X - np.mean(Args.X, axis=1, keepdims=True)

    # Normalize to norm 1 for cosine and correlation
    if Args.similarity in ["cosim", "corr"]:
        Args.X = Args.X / np.linalg.norm(Args.X, axis=1, keepdims=True)
    elif Args.similarity in ["dot", "cov"]:
        Args.X = Args.X / np.sqrt(Args.p)

    # Compute self-connection weights
    if Args.similarity == "network":
        Args.Wii = np.diag(Args.W).T
    else:
        Args.Wii = np.sum(Args.X**2, axis=1).T

    # Precompute kmeans++ variables
    Args.Dist = np.array([])
    Args.normX = np.array([])
    if Args.start in ["greedy", "balanced"]:
        if Args.similarity == "network":
            Args.Dist = Args.W / np.linalg.norm(Args.W, axis=1, keepdims=True)
            Args.Dist = 1 - Args.Dist @ Args.Dist.T
        else:
            Args.normX = np.linalg.norm(Args.X, axis=1)

    return Args
