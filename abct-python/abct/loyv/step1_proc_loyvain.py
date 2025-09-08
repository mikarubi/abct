import abct
import numpy as np

def step1_proc_loyvain(Args):
    # Loyvain arguments processing

    Args.n, Args.p = Args.X.shape
    if Args.similarity == "network":
        Args.W = Args.X
        Args.X = np.array(None)
    else:
        Args.W = np.array(None)

    # Process custom initial module assignment
    if isinstance(Args.start, np.ndarray):
        Args.M0 = np.ravel(Args.start).astype(int)
        if Args.k == 0:
            Args.k = np.max(Args.M0) + 1
        Args.start = "custom"

    # Global residualization for kmodularity
    if Args.objective == "kmodularity":
        if Args.similarity == "network":
            Args.W = Args.W * (Args.n / Args.k) / np.sum(np.abs(Args.W))
            Args.W = abct.residualn(Args.W, "degree")
        else:
            Args.X = abct.residualn(Args.X, "global")
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
        Args.Wii = np.diag(Args.W)
    else:
        Args.Wii = np.sum(Args.X**2, axis=1)

    # Precompute kmeans++ variables
    Args.Dist, Args.normX = [0], [0]
    if Args.start in ["greedy", "balanced"]:
        if Args.similarity == "network":
            Args.Dist = Args.W / np.linalg.norm(Args.W, axis=1, keepdims=True)
            Args.Dist = 1 - Args.Dist @ Args.Dist.T
        else:
            Args.normX = np.linalg.norm(Args.X, axis=1, keepdims=True)

    return Args
