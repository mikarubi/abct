function Args = step0_args(Args)
% Loyvain arguments initialization

arguments
    Args.X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
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

err = "Start must be either ""greedy"", ""balanced"", ""random"", or a numeric vector.";
assert(isnumeric(Args.start) || ismember(Args.start, ["greedy", "balanced", "random"]), err)

end
