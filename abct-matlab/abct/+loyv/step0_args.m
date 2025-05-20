function Args = step0_args(method, varargin)
% Loyvain arguments initialization

n_args = length(varargin);
assert(n_args >= 2, "Wrong number of input arguments.");
switch method
    case "loyvain"
        n_args_num = 2;
        [W, k] = deal(varargin{1:n_args_num});
        X = W;
        Y = 0;
    case "coloyvain"
        n_args_num = 2 + (n_args > 2 && isnumeric(varargin{3}));
        if n_args_num == 2
            [W, k] = deal(varargin{1:n_args_num});
            [X, Y] = deal(0);
        elseif n_args_num == 3
            [X, Y, k] = deal(varargin{1:n_args_num});
            W = 0;
        end
end
varargin = varargin(n_args_num+1:end);
if n_args >= n_args_num + 1
    varargin = [varargin(2:end), {"objective"}, varargin(1)];
    if n_args >= n_args_num + 2
        varargin = [varargin(2:end), {"similarity"}, varargin(1)];
    end
end

Args = parse_args("method", method, "W", W, "X", X, "Y", Y, "k", k, varargin{:});

end

function Args = parse_args(Args)

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
    Args.maxiter (1, 1) {mustBeInteger, mustBeNonnegative} = 1000
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
