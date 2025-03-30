function Args = step1_proc_loyvain(Args)
% Loyvain arguments processing

[Args.n, Args.t] = size(Args.X);
if Args.similarity == "network"
    Args.W = Args.X;
    Args.X = [];
else
    Args.W = [];
end

% Process custom initial module assignment
if isnumeric(Args.start)
    Args.M0 = Args.start;
    if Args.k == 0
        Args.k = max(Args.M0);
    end
    Args.start = "custom";
end

% Remove first mode for modularity
if Args.objective == "modularity"
    if Args.similarity == "network"
        Args.W = Args.W * (Args.n/Args.k) / sum(abs(Args.W), "all");
        Args.W = moderemoval(Args.W, "degree");
    else
        Args.X = moderemoval(Args.X, "global");
    end
    Args.objective = "kmeans";
end

% Center to mean 0 for covariance and correlation
if ismember(Args.similarity, ["cov", "corr"])
    Args.X = Args.X - mean(Args.X, 2);
end

% Normalize to norm 1 for cosine and correlation
if ismember(Args.similarity, ["cosim", "corr"])
    Args.X = Args.X ./ vecnorm(Args.X, 2, 2);
elseif ismember(Args.similarity, ["dot", "cov"])
    Args.X = Args.X / sqrt(Args.t);
end

% Compute self-connection weights
if Args.similarity == "network"
    Args.Wii = diag(Args.W)';
else
    Args.Wii = sum(Args.X.^2, 2)';
end

% Precompute kmeans++ variables
Args.Dist = [];
Args.normX = [];
if ismember(Args.start, ["greedy", "balanced"])
    if Args.similarity == "network"
        Args.Dist = Args.W ./ vecnorm(Args.W, 2, 2);
        Args.Dist = 1 - Args.Dist * Args.Dist';
    else
        Args.normX = vecnorm(Args.X, 2, 2);
    end
end

end
