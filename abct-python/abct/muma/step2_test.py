import numpy as np

def step2_test(Args):
    # m-umap arguments tests

    if Args.d >= Args.n:
        raise ValueError("Embedding dimension must be less than number of nodes or data points.")

    # Test partition
    if (len(Args.partition) != Args.n) or not np.array_equal(np.unique(Args.partition), np.arange(Args.k)):
        raise ValueError("Initial module partition must have length %d and contain integers 0 to %d." % (Args.n, Args.k - 1))

    # Test initial embedding
    if Args.start == "custom":
        if Args.U.shape != (Args.n, Args.d):
            raise ValueError("Initial embedding must have %d rows and %d columns." % (Args.n, Args.d))

    # Test initializations
    match Args.start:
        case "greedy":
            if Args.d != 3:
                raise ValueError("Embedding dimension must be 3 for ""greedy"" initialization.")
        case "spectral":
            if Args.d > Args.k:
                raise ValueError("Number of modules is too small for ""spectral"" initialization.")
        case "spectral_nn":
            if Args.d >= Args.n - 1:
                raise ValueError("Embedding dimension must be < n - 1 for ""spectral_nn"" initialization.")

