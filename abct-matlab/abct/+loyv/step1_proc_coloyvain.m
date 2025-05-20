function Args = step1_proc_coloyvain(Args)
% co-Loyvain arguments processing

if Args.similarity ~= "network"
    Args.X = proc(Args.X, Args);
    Args.Y = proc(Args.Y, Args);
    Args.W = Args.X' * Args.Y;
    Args.similarity = "network";
end
[Args.px, Args.py] = size(Args.W);

% Remove first mode for kmodularity
if Args.objective == "kmodularity"
    Args.W = Args.W * (sqrt(Args.px*Args.py)/Args.k) / sum(abs(Args.W), "all");
    Args.W = moderemoval(Args.W, "degree");
end
switch Args.objective
    case "kmodularity"; Args.objective = "cokmeans";
    case "kmeans";      Args.objective = "cokmeans";
    case "spectral";    Args.objective = "cospectral";
end
if ismember(Args.start, ["greedy", "balanced"])
    Args.DistX = Args.W ./ vecnorm(Args.W, 2, 2);
    Args.DistY = Args.W ./ vecnorm(Args.W, 2, 1);
    Args.DistX = 1 - (Args.DistX * Args.DistX');
    Args.DistY = 1 - (Args.DistY' * Args.DistY);
else
    [Args.DistX, Args.DistY] = deal(0);
end

end

function X = proc(X, Args)

% Center data points to mean 0 for covariance and correlation
if ismember(Args.similarity, ["cov", "corr"])
    X = X - mean(X, 1);
end

% Normalize data points to norm 1 for cosine and correlation
if ismember(Args.similarity, ["cosim", "corr"])
    X = X ./ vecnorm(X, 2, 1);
elseif ismember(Args.similarity, ["dot", "cov"])
    X = X / sqrt(size(X, 1));
end

end
