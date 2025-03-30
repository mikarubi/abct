function Args = step0_args(method, Args)
% Loyvain arguments initialization

arguments
    method (1, 1) = "loyvain";
    Args.X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    Args.Y (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    Args.k (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
    Args.objective (1, 1) string {mustBeMember(Args.objective, ...
        ["modularity", "kmeans", "spectral"])} = "modularity"
    Args.similarity (1, 1) string {mustBeMember(Args.similarity, ...
        ["network", "corr", "cosim", "cov", "dot"])} = "network"
    Args.numbatches (1, 1) double {mustBeInteger, mustBePositive} = 2
    Args.maxiter (1, 1) {mustBeInteger, mustBePositive} = 1000
    Args.replicates (1, 1) {mustBeInteger, mustBePositive} = 10
    Args.start (1, :) = "greedy"
    Args.display (1, 1) string {mustBeMember(Args.display, ...
        ["none", "replicate", "iteration"])} = "none"
end

Args.method = method;
if (Args.method == "loyvain") && isnumeric(Args.start)
elseif (isStringScalar(Args.start) || ischar(Args.start)) && ...
        ismember(Args.start, ["greedy", "balanced", "random"])
else
    error("Start must be either ""greedy"", ""balanced"", " + ...
        """random"", or a numeric vector for loyvain.");
end
if (Args.method == "coloyvain")
    assert(size(Args.X, 1) == size(Args.Y, 1), "X and Y must have the same number of rows.")
    assert(Args.similarity ~= "network", "Network similarity is incompatible with coloyvain.")
end

end
