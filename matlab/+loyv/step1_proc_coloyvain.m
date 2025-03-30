function Args = step1_proc_coloyvain(Args)
% co-Loyvain arguments processing

[Args.X, Args.px, Args.s, Args.Cx, Args.DistX] = proc(Args.X, Args);
[Args.Y, Args.py, Args.s, Args.Cy, Args.DistY] = proc(Args.Y, Args);
Args.W = Args.X * Args.Y';

if Args.objective == "modularity"
    Args.objective = "kmeans";
end

end

function [X, p, s, C, DistC] = proc(X, Args)

[p, s] = size(X);

% Remove first mode for modularity
if Args.objective == "modularity"
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

% Precompute kmeans++ variables
C = X * X';
DistC = [];
if ismember(Args.start, ["greedy", "balanced"])
    DistC = C ./ vecnorm(C, 2, 2);
    DistC = 1 - DistC * DistC';
end

end
