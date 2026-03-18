function step2_test(X, W, n, k, Args)
% Loyvain arguments tests

assert(k >= 1, "Specify number of modules or starting module assignment.")
assert(k <= n, "Number of modules must be not exceed number of nodes or data points.")
assert(Args.numbatches <= n, "Number of batches must not exceed number of nodes or data points.")
assert(all(isfinite(X), "all"), "Data matrix has non-finite elements after processing.")

% Test data non-negativity for spectral clustering
if ismember(Args.objective, ["spectral", "cospectral"]) && (Args.similarity ~= "network")
    ending = "non-negative for ""spectral"" objective.";
    if numel(X) < 1e6
        assert(all(X * X' >= 0, "all"), "Similarity matrix must be " + ending)
    else
        warning("Not checking similarity matrix for negative values because " + ...
            "of large data size. Ensure that similarity matrix is " + ending)
    end
end

if Args.method == "loyvain"
    % Test symmetry
    assert(isequal(size(W, 1), size(W, 2)) && all(abs(W - W') < double(eps("single")), "all"), ...
        "Network matrix must be symmetric or similarity must not be ""network"".")

    % Test initialization
    if Args.start == "custom"
        message = sprintf(...
            "Initial module assignment must have length %d and contain integers 1 to %d.", n, k);
        if ismember(Args.objective, ["modularity", "modularity_ctr1", "modularity_ctr2"])
            assert((length(Args.M0) == n) && isempty(setdiff(unique(Args.M0), 1:k)), message)
        else
            assert((length(Args.M0) == n) && isequal(unique(Args.M0), 1:k), message)
        end
    end

end
