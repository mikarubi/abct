function step2_test(Args)
% Loyvain arguments tests

% Unpack arguments
X = Args.X;
W = Args.W;
k = Args.k;
n = Args.n;
t = Args.t;

assert(k >= 1, "Specify number of modules or starting module assignment.")
assert(k <= n, "Number of modules must be smaller than number of nodes.")
assert(Args.numbatches <= n, "Number of batches must not be larger than number of nodes.")
assert(all(isfinite(X), "all"), "Data matrix must be finite and not have zero rows.")
assert(isequal(size(W, 1), size(W, 2)) && all(W - W' < eps("single"), "all"), ...
    "Network matrix must be symmetric or similarity must not be ""network"".")

% Test non-negativity for spectral and modularity
if ismember(Args.objective, ["modularity" "spectral"])
    if Args.similarity == "network"
        assert(all(W >= 0, "all"), "Network matrix must be non-negative.");
    elseif (Args.objective == "spectral") && (n*t < 1e6)
        assert(all(X * X' >= 0, "all"), "Similarity matrix must be non-negative.")
    elseif (Args.objective == "spectral")
        warning("Not checking similarity matrix for negative values because " + ...
            "of large data size. Ensure that similarity matrix is non-negative.")
    end
end

% Test initialization
if Args.start == "custom"
    assert((length(Args.M0) == n) && isequal(unique(Args.M0), 1:k), ...
        "Initial module assignment must have length %d and contain integers 1 to %d.", n, k)
end

end
