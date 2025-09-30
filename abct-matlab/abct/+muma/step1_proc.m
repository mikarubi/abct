function Args = step1_proc(Args)
% m-umap arguments processing

% Process custom initial embedding
if isnumeric(Args.start)
    Args.U = Args.start;
    Args.start = "custom";
end

% Generate a nearest-neighbor matrix
if Args.similarity == "none"
    if Args.verbose
        warning("Similarity is ""none"". Ignoring ""kappa"" and ""method"" arguments.")
    end
    Args.A = double(Args.X);
    Args.A(1:length(Args.A)+1:end) = 0;
else
    Args.A = kneighbor(Args.X, ...
        "nearest", ...
        Args.kappa, ...
        Args.similarity, ...
        Args.method);
end

% Module structure
if isempty(Args.partition)
    switch Args.verbose
        case true; display = "replicate";
        case false; display = "none";
    end
    Args.partition = louvains(Args.A, ...
        gamma=Args.gamma, ...
        replicates=Args.replicates, ...
        finaltune=Args.finaltune, ...
        display=display);
else
    [~, ~, Args.partition] = unique(Args.partition);
end

Args.n = size(Args.X, 1);
Args.k = max(Args.partition);
Args.M = full(sparse(1:Args.n, Args.partition, 1));
Args.Am = Args.A * Args.M;

end
