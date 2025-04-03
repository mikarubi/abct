function Args = step0_args(Args)
% Loyvain arguments initialization

arguments
    Args.method
    Args.X
    Args.Y
    Args.k
    Args.objective
    Args.similarity
    Args.start (1, :) = "greedy"
    Args.numbatches (1, 1) double {mustBeInteger, mustBePositive} = 2
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
if (Args.method == "coloyvain")
    assert(size(Args.X, 2) == size(Args.Y, 2), ...
    "X and Y must have the same number of observations.")
end

end
