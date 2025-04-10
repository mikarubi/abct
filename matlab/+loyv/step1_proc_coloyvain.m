function Args = step1_proc_coloyvain(Args)
% co-Loyvain arguments processing

[Args.X, Args.Wx, Args.Wx_ii, Args.DistX, Args.px, Args.s] = proc(Args.X, Args);
[Args.Y, Args.Wy, Args.Wy_ii, Args.DistY, Args.py, Args.s] = proc(Args.Y, Args);
Args.Wxy = Args.X * Args.Y';

switch Args.objective
    case "kmodularity"; Args.objective = "cokmeans";     % first mode already removed
    case "kmeans";      Args.objective = "cokmeans";
    case "spectral";    Args.objective = "cospectral";
end

end

function [X, W, Wii, Dist, p, s] = proc(X, Args)

[p, s] = size(X);

% Remove first mode for kmodularity
if Args.objective == "kmodularity"
    X = moderemoval(X, "global");
end

% Center to mean 0 for covariance and correlation
if ismember(Args.similarity, ["cov", "corr"])
    X = X - mean(X, 2);
end

% Normalize to norm 1 for cosine and correlation
if ismember(Args.similarity, ["cosim", "corr"])
    X = X ./ vecnorm(X, 2, 2);
elseif ismember(Args.similarity, ["dot", "cov"])
    X = X / sqrt(s);
end

% Compute correlation weights
W = X * X';
Wii = diag(W)';

% Precompute kmeans++ variables
Dist = [];
if ismember(Args.start, ["greedy", "balanced"])
    Dist = W ./ vecnorm(W, 2, 2);
    Dist = 1 - Dist * Dist';
end

end
