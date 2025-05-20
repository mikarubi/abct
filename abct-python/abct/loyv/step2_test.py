import warnings

import numpy as np

def step2_test(X, W, n, k, Args):
    # Loyvain arguments tests

    if k < 1:
        raise ValueError("Specify number of modules or starting module assignment.")
    if k > n:
        raise ValueError("Number of modules must not exceed number of nodes or data points.")
    if Args.numbatches > n:
        raise ValueError("Number of batches must not exceed number of nodes or data points.")
    if not (np.array_equal(X, None) or np.all(np.isfinite(X))):
        raise ValueError("Data matrix has non-finite elements after processing.")

    # Test non-negativity for spectral clustering
    if Args.objective in ["spectral", "cospectral"]:
        ending = 'non-negative for "spectral" objective.'
        if Args.similarity == "network":
            if not np.all(W >= 0):
                raise ValueError("Network matrix must be " + ending)
        elif X.size < 1e6:
            if not np.all(X @ X.T >= 0):
                raise ValueError("Similarity matrix must be " + ending)
        else:
            warnings.warn(
                "Not checking similarity matrix for negative values because "
                "of large data size. Ensure that similarity matrix is " + ending
            )

    if Args.method == "loyvain":
        # Test symmetry
        if not ((W.shape[0] == W.shape[1]) and np.allclose(W, W.T)):
            raise ValueError('Network matrix must be symmetric or similarity must not be "network".')

        # Test initialization
        if Args.start == "custom":
            if not ((len(Args.M0) == n) and np.array_equal(np.unique(Args.M0), np.arange(k))):
                raise ValueError(f"Initial module assignment must have length {n} and contain integers {0} to {k-1}.")
