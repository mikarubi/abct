function Args = step0_args(X, Args)
% m-umap arguments initialization

arguments
    X (:, :) {mustBeFinite, mustBeReal}
    Args.d (1, 1) {mustBePositive, mustBeInteger} = 3
    Args.kappa (1, 1) {mustBePositive} = 30
    Args.alpha (1, 1) {mustBePositive} = 1
    Args.beta (1, 1) {mustBePositive} = 1
    Args.gamma (1, 1) {mustBePositive} = 1
    Args.similarity (1, 1) string {mustBeMember( ...
        Args.similarity, ["network", "corr", "cosim"])} = "network"
    Args.method (1, 1) string {mustBeMember( ...
        Args.method, ["direct", "indirect"])} = "direct"
    Args.replicates (1, 1) {mustBePositive, mustBeInteger} = 10
    Args.finaltune (1, 1) logical = true
    Args.partition (:, 1) {mustBePositive, mustBeInteger} = []
    Args.start (1, :) = "greedy"
    Args.solver (1, 1) string {mustBeMember( ...
        Args.solver, ["adam", "trustregions"])} = "trustregions"
    Args.maxiter (1, 1) {mustBePositive, mustBeInteger} = 1e4
    Args.learnrate (1, 1) {mustBePositive} = 1e-3
    Args.tolerance (1, 1) {mustBePositive} = 1e-6
    Args.gpu (1, 1) logical = false
    Args.cache (1, 1) logical = false
    Args.verbose (1, 1) logical = true
end
Args.X = X;

if Args.similarity == "network"
    % Test symmetry
    assert(isequal(size(Args.X, 1), size(Args.X, 2)) && ...
        all(abs(Args.X - Args.X') < eps("single"), "all"), ...
        "Network matrix must be symmetric or similarity must not be ""network"".")
end

if Args.gpu
    assert(license("test", "distrib_computing_toolbox") && gpuDeviceCount, ...
        "GPU use requires Parallel Computing Toolbox and active GPU.")
end
switch Args.solver
    case "adam"
        assert(license("test", "neural_network_toolbox"), ...
            "Adam solver requires Deep Learning Toolbox.")
    case "trustregions"
        assert((exist("obliquefactory", "file") && exist("trustregions", "file")), ...
            "Manopt functions not found. Download manopt and call importmanopt.")
end
if isnumeric(Args.start)
elseif (isStringScalar(Args.start) || ischar(Args.start)) && ...
    ismember(Args.start, ["greedy", "spectral", "spectral_nn"])
else
    error("Start must be either ""greedy"", ""spectral"", " + ...
        """spectral_nn"", or a numeric matrix.");
end

end
