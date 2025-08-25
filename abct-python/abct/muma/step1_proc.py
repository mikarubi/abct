import abct
import numpy as np

def step1_proc(Args):
    # m-umap arguments processing

    # Process custom initial embedding
    if isinstance(Args.start, np.ndarray):
        Args.U = Args.start
        Args.start = "custom"

    Args.n = len(Args.X)
    if Args.similarity == "network":
        Args.A = Args.X
        np.fill_diagonal(Args.A, 0)
    else:
        # Generate a nearest-neighbor matrix
        Args.A = abct.kneighbors(Args.X, "nearest", Args.kappa, Args.similarity, Args.method)

    # Module structure
    if Args.partition is None:
        Args.partition = abct.louvains(Args.A,
            gamma=Args.gamma,
            replicates=Args.replicates,
            finaltune=Args.finaltune
        )
    else:
        Args.partition = np.unique_inverse(Args.partition).inverse_indices

    Args.k = np.max(Args.partition)
    Args.M = np.zeros((Args.n, Args.k))
    Args.M[np.r_[:Args.n], Args.partition] = 1
    Args.Am = Args.A @ Args.M

    return Args
