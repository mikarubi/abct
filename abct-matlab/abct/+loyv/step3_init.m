function M0 = step3_init(X, Norm, Dist, n, Args)
% Loyvain output initialization

% Unpack arguments
k = Args.k;
if Args.similarity == "network"
    Mean = mean(Dist, 2);
else
    Mean = [];
end

if ismember(Args.start, ["greedy", "balanced"])
    Idx = [randi(n) nan(1, k-1)];               % centroid indices
    minDist = inf(1, n);
    for j = 2:k
        % Distance from preceding centroid to all other nodes
        Dj = distance(X, Norm, Mean, Dist, Idx(j-1), Args);
        minDist = min(minDist, Dj);             % min distance to centroid
        if Args.start == "greedy"
            sampleProbability = (minDist == max(minDist));
        elseif Args.start == "balanced"
            sampleProbability = (minDist / sum(minDist));
        end
        P = [0 cumsum(sampleProbability)];
        P(end) = 1;
        Idx(j) = find(rand < P, 1) - 1;         % sample next centroid
    end
    % Distance from all centroids to all nodes
    D = distance(X, Norm, Mean, Dist, Idx, Args);
    [~, M0] = min(D, [], 1);                    % closest centroid to each node
    k0 = max(M0);
    if k0 < k
        M0(randperm(n, k-k0)) = k0+1:k;         % ensure there are k modules
    end
elseif Args.start == "random"
    M0 = randi(k, 1, n);                        % initial module partition
    M0(randperm(n, k)) = 1:k;                   % ensure there are k modules
end

end

function D = distance(X, Norm, Mean, Dist, Idx, Args)
if Args.method == "coloyvain"
    % Dist here is the precomputed distance
    D = Dist(Idx, :);
elseif Args.similarity == "network"
    % Dist here is the normalized network in step1
    DistC_Idx = Dist(Idx, :) - Mean(Idx);
    D = 1 - (DistC_Idx * Dist' - sum(DistC_Idx, 2) * Mean') ./ (Norm(Idx) * Norm');
else
    % X here is the data matrix
    D = 1 - (X(Idx, :) * X') ./ (Norm(Idx) * Norm');
end
end
