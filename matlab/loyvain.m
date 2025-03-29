function [M, Q] = loyvain(X, k, objective, args)
% LOYVAIN Normalized modularity, k-means, or spectral clustering
%
%   [M, Q] = loyvain(X, k)
%   [M, Q] = loyvain(X, k, objective)
%   [M, Q] = loyvain(X, k, objective, Name=Value)
%
%   Inputs:
%       X: Network matrix of size n x n, or data matrix of size n x t.
%          n is the number of nodes and t is the number of observations.
%
%       k: Number of modules (positive integer or 0).
%           Set to 0 to infer number from initial module assignment.
%
%       objective: Clustering objective.
%           "modularity": Normalized modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Spectral clustering objective (normalized cut).
%
%       Name=[Value] Arguments:
%
%           Similarity=[Type of similarity].
%               The default option assumes that X is a network matrix.
%                   "network": Network connectivity (default).
%                       X is a symmetric network matrix. The network must
%                       be non-negative for the spectral and modularity
%                       objectives. No additional similarity is computed.
%               The remaining options assume that X is a data matrix.
%                   "corr": Pearson correlation coefficient.
%                       A scale-invariant measure of linear association,
%                       a normalized dot product of mean-centered vectors.
%                   "cosim": Cosine similarity.
%                       A normalized dot product.
%                   "cov":  Covariance.
%                       A dot product of mean-centered vectors.
%                   "dot": Dot product.
%                       A sum of an elementwise vector product.
%
%           NumBatches=[Number of batches].
%               Positive integer (default is 2).
%
%           MaxIter=[Maximum number of algorithm iterations].
%               Positive integer (default is 1000).
%
%           Replicates=[Number of replicates].
%               Positive integer (default is 10).
%
%           Start=[Initial module assignments].
%               "greedy": Maximin (maximally greedy kmeans++) initialization (default).
%               "balanced": Standard kmeans++ initialization.
%               "random": Uniformly random initialization.
%               Initial-module-assignment vector of length n.
%
%           Display=[Display progress].
%               "none": no display (default).
%               "replicate": display progress at each replicate.
%               "iteration": display progress at each iteration.
%
%   Outputs:
%       M: Vector of module assignments (length n).
%
%       Q: Value of normalized modularity, k-means, or spectral objective.
%
%   Methodological notes:
%       Loyvain is a unification of:
%           Lloyd's algorithm for k-means clustering and
%           Louvain algorithm for modularity maximization.
%
%       Note 1. The normalized modularity maximization is equivalent to
%       k-means clustering of data after degree correction. When the input
%       is a data rather than a network matrix, degree correction is
%       implemented via an approximately equivalent step of global-signal
%       regression. More generally, degree correction and global-signal
%       regression are both approximately equivalent to first-mode removal,
%       or subtraction of the rank-one approximation of the data.
%
%       Note 2. For Similarity="network", the value of the normalized
%       modularity is rescaled by the following factor:
%           (average module size) / (absolute sum of all weights)
%       This rescaling approximately aligns the value of the objective
%       function with values of the unnormalized modularity. For other
%       similarity measures the value of the objective function is not
%       rescaled, but the algorithm optimizes an equivalent objective.
%
%       Note 3. The Loyvain algorithm is not guaranteed to converge if
%       all swaps are accepted at each iteration (NumBatches = 1).
%       Therefore, it is generally a good idea to set NumBatches > 1.
%
%   See also:
%       GRADIENTS, MODEREMOVAL.

arguments
    X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    k (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
    objective (1, 1) string {mustBeMember(objective, ...
        ["modularity", "kmeans", "spectral"])} = "modularity"
    args.similarity (1, 1) string {mustBeMember(args.similarity, ...
        ["network", "corr", "cosim", "cov", "dot"])} = "network"
    args.numbatches (1, 1) double {mustBeInteger, mustBePositive} = 2
    args.maxiter (1, 1) {mustBeInteger, mustBePositive} = 1000
    args.replicates (1, 1) {mustBeInteger, mustBePositive} = 10
    args.start (1, :) = "greedy"
    args.display (1, 1) string {mustBeMember(args.display, ...
        ["none", "replicate", "iteration"])} = "none"
end

%% Initial processing

[n, t] = size(X);
if args.similarity == "network"
    W = X;
    X = [];
else
    W = [];
end

% Remove first mode for modularity
if objective == "modularity"
    if args.similarity == "network"
        W = W * (n/k) / sum(abs(W), "all");
        W = moderemoval(W, "degree");
    else
        X = moderemoval(X, "global");
    end
    objective = "kmeans";
end

% Process custom initial module assignment
if isnumeric(args.start)
    M0 = args.start;
    if k == 0
        k = max(M0);
    end
    args.start = "custom";
end

% Center to mean 0 for covariance and correlation
if ismember(args.similarity, ["cov", "corr"])
    X = X - mean(X, 2);
end

% Normalize to norm 1 for cosine and correlation
if ismember(args.similarity, ["cosim", "corr"])
    X = X ./ vecnorm(X, 2, 2);
elseif ismember(args.similarity, ["dot", "cov"])
    X = X / sqrt(t);
end

% Compute self-connection weights
if args.similarity == "network"
    Wii = diag(W)';
else
    Wii = sum(X.^2, 2)';
end

%% Additional tests

assert(k >= 1, "Specify number of modules or starting module assignment.")
assert(k <= n, "Number of modules must be smaller than number of nodes.")
assert(args.numbatches <= n, "Number of batches must not be larger than number of nodes.")
assert(all(isfinite(X), "all"), "Data matrix must be finite and not have zero rows.")
assert(isequal(size(W, 1), size(W, 2)) && all(W - W' < eps("single"), "all"), ...
    "Network matrix must be symmetric or similarity must not be ""network"".")

% Test non-negativity for spectral and modularity
if ismember(objective, ["modularity" "spectral"])
    if args.similarity == "network"
        assert(all(W >= 0, "all"), "Network matrix must be non-negative.");
    elseif (objective == "spectral") && (n*t < 1e6)
        assert(all(X * X' >= 0, "all"), "Similarity matrix must be non-negative.")
    elseif (objective == "spectral")
        warning("Not checking similarity matrix for negative values because " + ...
            "of large data size. Ensure that similarity matrix is non-negative.")
    end
end

% Test initialization
assert(ismember(args.start, ["greedy", "balanced", "random", "custom"]), ...
    "Start must be either ""greedy"", ""balanced"", ""random"", or a numeric vector.")
if args.start == "custom"
    assert((length(M0) == n) && isequal(unique(M0), 1:k), ...
        "Initial module assignment must have length %d and contain integers 1 to %d.", n, k)
end

%% Run algorithm

% Precompute kmeans++ variables
if ismember(args.start, ["greedy", "balanced"])
    if args.similarity == "network"
        Dist = W ./ vecnorm(W, 2, 2);
        Dist = 1 - Dist * Dist';
    else
        normX = vecnorm(X, 2, 2);
    end
end

Q = - inf;
for i = 1:args.replicates
    if ismember(args.start, ["greedy", "balanced"])
        Idx = [randi(n) nan(1, k-1)];           % centroid indices
        minDist = inf(1, n);
        for j = 2:k
            if args.similarity == "network"     % use precomputed distance
                Dj = Dist(Idx(j-1), :);
            else                                % compute distance on the fly
                Dj = 1 - (X(Idx(j-1), :) * X') ./ (normX(Idx(j-1)) * normX');
            end
            minDist = min(minDist, Dj);         % min distance to centroid
            if args.start == "greedy"
                sampleProbability = (minDist == max(minDist));
            elseif args.start == "balanced"
                sampleProbability = (minDist / sum(minDist));
            end
            P = [0 cumsum(sampleProbability)]; P(end) = 1;
            Idx(j) = find(rand < P, 1) - 1;     % sample new centroid
        end
        if args.similarity == "network"         % use precomputed distance
            [~, M0] = min(Dist(Idx, :), [], 1);
        else                                    % compute distance on the fly
            [~, M0] = min(1 - (X(Idx, :) * X') ./ (normX(Idx) * normX'), [], 1);
        end
    elseif args.start == "random"
        M0 = randi(k, 1, n);                    % initial module partition
        M0(randperm(n, k)) = 1:k;               % ensure there are k modules
    end
    [M1, Q1] = run_loyvain(M0, X, W, Wii, n, k, objective, args, i);
    if Q1 > Q
        if ismember(args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, Q1, Q1 - Q);
        end
        Q = Q1;
        M = M1;
    end
end

end

function [M, Q] = run_loyvain(M, X, W, Wii, n, k, objective, args, replicate_i)

LinIdx = M + k*(0:n-1);                     % two-dimensional indices of M

MM = sparse(M, 1:n, 1, k, n);               % two-dimensional representation
N = full(sum(MM, 2));                       % number of nodes in module
if args.similarity == "network"
    Smn = MM * W;                           % degree of module to node
else
    G = MM * X;                             % cluster centroid
    Smn = G * X';                           % dot of centroid with node
end
Cii = diag(Smn * MM');                      % within-module weight sum

switch objective
    case "kmeans"
        Cii_nrm = Cii ./ N;
    case "spectral"
        Sn = sum(Smn, 1);                   % degree of node
        Sm = sum(Smn, 2);                   % degree of module
        Cii_nrm = Cii ./ Sm;
end

tol = 1e-10;
for v = 1:args.maxiter

    max_delta_Q = 0;                            % maximal increase over all batches
    Batches = mat2cell(randperm(n), 1, diff(round(0:n/args.numbatches:n)));
    for u = 1:args.numbatches
        U = Batches{u};                         % indices of nodes in batch
        LinU = LinIdx(U);                       % linear indices of nodes in batch
        MU = M(U);                              % module assignments of nodes in batch
        b = numel(U);                           % number of nodes in batch

        switch objective
            case "kmeans"
                delta_QU = ...
                    ((2 * Smn(:, U) + Wii(U)) - Cii_nrm     ) ./ (N      + 1) - ...
                    ((2 * Smn(LinU) - Wii(U)) - Cii_nrm(MU)') ./ (N(MU)' - 1);
            case "spectral"
                delta_QU = ...
                    ((2 * Smn(:, U) + Wii(U)) - Cii_nrm      .* Sn(U)) ./ (Sm      + Sn(U)) - ...
                    ((2 * Smn(LinU) - Wii(U)) - Cii_nrm(MU)' .* Sn(U)) ./ (Sm(MU)' - Sn(U));
        end
        delta_QU(:, N(MU) == 1) = - inf;        % no change allowed if one-node cluster
        delta_QU(MU + k*(0:b-1)) = 0;           % no change if node stays in own module

        % Update if improvements
        [max_delta_QU, MU_new] = max(delta_QU);
        if max(max_delta_QU) > tol
            max_delta_Q = max(max_delta_Q, max(max_delta_QU));

            IU = find(MU ~= MU_new);            % batch indices of nodes to be switched
            I = U(IU);                          % actual indices of nodes to be switched
            MI_new = MU_new(IU);                % new module assignments

            % get delta modules and ensure non-empty modules
            n_i = numel(I);
            MMI = sparse(M(I), 1:n_i, 1, k, n_i);
            while 1
                MMI_new = sparse(MI_new, 1:n_i, 1, k, n_i);
                delta_MMI = (MMI_new - MMI);
                N_new = N + sum(delta_MMI, 2);
                if all(N_new)
                    break;
                else
                    E = find(~N_new);           % empty modules
                    k_e = numel(E);             % number of empty modules
                    MI_new(randperm(n_i, k_e)) = E;
                end
            end

            % Update N, M, MM, and LinIdx
            N = N_new;
            M(I) = MI_new;
            MM = sparse(M, 1:n, 1, k, n);
            LinIdx(I) = MI_new + k*(I-1);

            % Update G and Dmn
            if args.similarity == "network"
                delta_Dmn = delta_MMI * W(I, :);
            else
                delta_G = delta_MMI * X(I, :);  % change in centroid
                G = G + delta_G;                % update centroids
                delta_Dmn = delta_G * X';       % change in degree of module to node
            end
            Smn = Smn + delta_Dmn;              % update degree of module to node

            % Get Cii, C_nrm, and Dm
            Cii = diag(Smn * MM');              % within-module weight sum
            switch objective
                case "kmeans"
                    Cii_nrm = Cii ./ N;
                case "spectral"
                    Sm = Sm + sum(delta_Dmn, 2);
                    Cii_nrm = Cii ./ Sm;
            end
        end
    end
    if max_delta_Q < tol
        break
    end
    if args.display == "iteration"
        fprintf("Replicate: %4d.    Iteration: %4d.    Largest \x0394: %4.4f\n", ...
            replicate_i, v, max_delta_Q)
    end
    if v == args.maxiter
        warning("Algorithm did not converge after %d iterations.", v)
    end
end

% Return objective
Q = sum(Cii_nrm);

end
