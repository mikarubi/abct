import abct
import numpy as np

def step1_proc_coloyvain(Args):
    # co-Loyvain arguments processing

    if Args.similarity != "network":
        Args.X = proc(Args.X, Args)
        Args.Y = proc(Args.Y, Args)
        Args.W = Args.X.T @ Args.Y
        Args.similarity = "network"

    Args.px, Args.py = Args.W.shape

    # Global residualization for kmodularity
    if Args.objective == "kmodularity":
        Args.W = Args.W * (np.sqrt(Args.px * Args.py) / Args.k) / np.sum(np.abs(Args.W))
        Args.W = abct.residualn(Args.W, "degree")

    match Args.objective:
        case "kmodularity": Args.objective = "cokmeans"
        case "kmeans":      Args.objective = "cokmeans"
        case "spectral":    Args.objective = "cospectral"

    if Args.start in ["greedy", "balanced"]:
        Args.DistX = Args.W / np.linalg.norm(Args.W, axis=1, keepdims=True)
        Args.DistY = Args.W / np.linalg.norm(Args.W, axis=0, keepdims=True)
        Args.DistX = 1 - (Args.DistX @ Args.DistX.T)
        Args.DistY = 1 - (Args.DistY.T @ Args.DistY)
    else:
        Args.DistX, Args.DistY = [0], [0]

    return Args


def proc(X, Args):

    # Center data points to mean 0 for covariance and correlation
    if Args.similarity in ["cov", "corr"]:
        X = X - np.mean(X, axis=0)

    # Normalize data points to norm 1 for cosine and correlation
    if Args.similarity in ["cosim", "corr"]:
        X = X / np.linalg.norm(X, axis=0, keepdims=True)
    elif Args.similarity in ["dot", "cov"]:
        X = X / np.sqrt(len(X))

    return X
