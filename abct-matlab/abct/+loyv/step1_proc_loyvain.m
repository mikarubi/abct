function Args = step1_proc_loyvain(Args)
% Loyvain arguments processing

[Args.n, Args.p] = size(Args.X);
if Args.similarity == "network"
    Args.W = Args.X;
    Args.X = 0;
else
    Args.W = 0;
end

% Process custom initial module assignment
if isnumeric(Args.start)
    Args.M0 = Args.start;
    if Args.k == 0
        Args.k = max(Args.M0);
    end
    Args.start = "custom";
end

% Normalization and residualization for kmodularity
if ismember(Args.objective, ...
        ["kmodularity", "modularity", "modularity_ctr1", "modularity_ctr2"])
    if Args.similarity == "network"
        % residualize in step4 to avoid constructing full n * n matrices
        Args.W = Args.W / sum(abs(Args.W), "all");
        if Args.objective == "kmodularity"
            Args.W = Args.W * (Args.n / Args.k);
        end
    else
        % residualize here to enable subsequent centering and normalization
        if ismember(Args.objective, ["kmodularity", "modularity"])
            Args.X = residualn(Args.X, "global");
        elseif Args.objective == "modularity_ctr2"
            Args.X = residualn(Args.X, "global_ctr");
        end
    end
end

% Center to mean 0 for covariance and correlation
if ismember(Args.similarity, ["cov", "corr"])
    Args.X = Args.X - mean(Args.X, 2);
end

% Normalize to norm 1 for cosine and correlation
if ismember(Args.similarity, ["cosim", "corr"])
    Args.X = Args.X ./ vecnorm(Args.X, 2, 2);
elseif ismember(Args.similarity, ["dot", "cov"])
    Args.X = Args.X / sqrt(Args.p);
end

% Precompute kmeans++ variables
% One-pass norm-rescaling acts as a rough degree correction that both
% maintains sparsity and also naturally works with negative values.
[Args.Dist, Args.Norm] = deal(0);
if ismember(Args.start, ["greedy", "balanced"])
    if Args.similarity == "network"
        S = sqrt(sqrt(sum(Args.W.^2, 2)));          % sqrt(vecnorm(W)) for geomean
        S(~S) = 1;                                  % protect isolated nodes
        Args.Dist = (Args.W ./ S) ./ S';            % efficient norm rescaling
        % centered norm
        Args.Norm = sqrt(sum(Args.Dist.^2, 2) - Args.n * mean(Args.Dist, 2).^2);
    else
        Args.Norm = vecnorm(Args.X, 2, 2);
    end
end
Args.Norm(~Args.Norm) = 1;

end
