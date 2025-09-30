import abct
import numpy as np
import warnings

def step1_proc(Args):
    # m-umap arguments processing

    # Process custom initial embedding
    if isinstance(Args.start, np.ndarray):
        Args.U = Args.start
        Args.start = "custom"

    # Generate a nearest-neighbor matrix
    if Args.similarity == "none":
        if Args.verbose:
            warnings.warn('Similarity is "none". Ignoring "kappa" and "method" arguments.')
        Args.A = np.asarray(Args.X)
        np.fill_diagonal(Args.A, 0)
    else:
        Args.A = abct.kneighbor(Args.X,
            "nearest",
            Args.kappa,
            Args.similarity,
            Args.method,
        )

    # Module structure
    if Args.partition is None:
        Args.partition = abct.leiden(Args.A,
            gamma=Args.gamma,
            replicates=Args.replicates,
            finaltune=Args.finaltune,
            display="replicate" if Args.verbose else "none",
        )[0]
    else:
        Args.partition = np.unique_inverse(Args.partition).inverse_indices

    Args.n = len(Args.X)
    Args.k = np.max(Args.partition) + 1
    Args.M = np.zeros((Args.n, Args.k))
    Args.M[np.arange(Args.n), Args.partition] = 1
    Args.Am = Args.A @ Args.M

    return Args
