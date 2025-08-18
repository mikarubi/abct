function Args = step1_proc(Args)
% m-umap arguments processing

% Process custom initial embedding
if isnumeric(Args.start)
    Args.U = Args.start;
    Args.start = "custom";
end

Args.n = size(Args.X, 1);
if Args.similarity == "network"
    Args.A = Args.X;
    Args.A(1:Args.n+1:end) = 0;
else
    % Generate a nearest-neighbor matrix
    Args.A = kneighbors(Args.X, ...
        "nearest", ...
        Args.kappa, ...
        Args.similarity, ...
        Args.method);
end

% Module structure
if isempty(Args.partition)
    Args.partition = louvains(Args.A, ...
        gamma=Args.gamma, ...
        replicates=Args.replicates, ...
        finaltune=Args.finaltune);
else
    [~, ~, Args.partition] = unique(Args.partition);
end

Args.k = max(Args.partition);
Args.M = full(sparse(1:Args.n, Args.partition, 1));
Args.Am = Args.A * Args.M;

end
