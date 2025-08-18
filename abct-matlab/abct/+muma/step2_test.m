function step2_test(Args)
% m-umap arguments tests

assert(Args.d < Args.n, "Embedding dimension must be less than number of nodes or data points.")

% Test partition
assert((length(Args.partition) == Args.n) && isequal(unique(Args.partition), 1:Args.k), ...
    "Initial module partition must have length %d and contain integers 1 to %d.", Args.n, Args.k)

% Test initial embedding
if Args.start == "custom"
    assert(isequal(size(Args.U), [Args.n, Args.d]), ...
        "Initial embedding must have %d rows and %d columns.", Args.n, Args.d)
end

% Test initializations
switch Args.start
    case "greedy"
        assert(Args.d == 3, "Embedding dimension must be 3 for ""greedy"" initialization.")
    case "spectral"
        assert(Args.d <= Args.k, "Number of modules is too small for ""spectral"" initialization.")
    case "spectral_nn"
        assert(Args.d < Args.n - 1, "Embedding dimension must be < n - 1 for ""spectral_nn"" initialization.")
end

end
