function Args = step0_args(Args)
% Loyvain arguments initialization

arguments
    Args.method (1, 1) string {mustBeMember(Args.method, ["loyvain", "coloyvain"])}
    Args.W (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite} = 0
    Args.X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite} = 0
    Args.Y (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite} = 0
    Args.k (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
    Args.objective (1, 1) string {mustBeMember(Args.objective, ...
        ["kmodularity", "kmeans", "spectral"])} = "kmodularity"
    Args.similarity (1, 1) string {mustBeMember(Args.similarity, ...
        ["network", "corr", "cosim", "cov", "dot"])} = "network"
    Args.start (1, :) = "greedy"
    Args.numbatches (1, 1) double {mustBeInteger, mustBePositive} = 10
    Args.maxiter (1, 1) {mustBeInteger, mustBePositive} = 1000
    Args.replicates (1, 1) {mustBeInteger, mustBePositive} = 10
    Args.tolerance (1, 1) double {mustBePositive} = 1e-10
    Args.display (1, 1) string {mustBeMember(Args.display, ...
        ["none", "replicate", "iteration"])} = "none"
end

if isnumeric(Args.start) && (Args.method == "loyvain")
elseif (isStringScalar(Args.start) || ischar(Args.start)) && ...
        ismember(Args.start, ["greedy", "balanced", "random"])
else
    error("Start must be either ""greedy"", ""balanced"", " + ...
        """random"", or a numeric vector (loyvain only).");
end
if Args.method == "coloyvain"
    assert(Args.k > 0, "k must be positive for co-Loyvain.")
    if Args.similarity == "network"
        assert(isequal(Args.X, 0) && isequal(Args.Y, 0), ...
            "X and Y inputs are incompatible with ""network"" similarity.")
    else
        assert(isequal(Args.W, 0), ...
            "W input is only compatible with ""network"" similarity.")
        assert(size(Args.X, 1) == size(Args.Y, 1), ...
            "X and Y must have the same number of data points.")
    end
end

end
