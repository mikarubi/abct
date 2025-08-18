function M0 = step3_init(X, normX, Dist, n, Args)
% Loyvain output initialization

% Unpack arguments
k = Args.k;

if ismember(Args.start, ["greedy", "balanced"])
    Idx = [randi(n) nan(1, k-1)];               % centroid indices
    minDist = inf(1, n);
    for j = 2:k
        if (Args.similarity == "network") || (Args.method == "coloyvain")
            % use precomputed distance
            Dj = Dist(Idx(j-1), :);
        else
            % compute distance on the fly
            Dj = 1 - (X(Idx(j-1), :) * X') ./ (normX(Idx(j-1)) * normX');
        end
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
    if (Args.similarity == "network") || (Args.method == "coloyvain")
        % use precomputed distance
        [~, M0] = min(Dist(Idx, :), [], 1);
    else
        % compute distance on the fly
        [~, M0] = min(1 - (X(Idx, :) * X') ./ (normX(Idx) * normX'), [], 1);
    end
    k0 = max(M0);
    if k0 < k
        M0(randperm(n, k-k0)) = k0+1:k;         % ensure there are k modules
    end
elseif Args.start == "random"
    M0 = randi(k, 1, n);                        % initial module partition
    M0(randperm(n, k)) = 1:k;                   % ensure there are k modules
end

end
