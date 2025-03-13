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
%                   "network": X is a symmetric network (default).
%                       No additional similarity is computed.
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
%           Acceptance=[Probability of acceptance of individual swaps].
%               0 < Acceptance < 1 (default is 0.5).
%               Higher values lead to faster convergence but also have
%               higher likelihood of getting stuck in local optima.
%
%           MaxIter=[Maximum number of algorithm iterations].
%               Positive integer (default is 1000).
%
%           Start=[Initial module assignments or number of starts].
%               Positive integer vector of length n.
%                   Initial module assignments.
%               Positive integer: (default is 10)
%                   Number of starts from module assignments
%                   initialized with the kmeans++ algorithm.
%
%           Verbose=[Display progress].
%               Logical (default is false).
%
%   Outputs:
%       M: Vector of module assignments (length n).
%
%       Q: Value of normalized modularity, k-means, or spectral objective.
%
%   Methodological notes:
%       Loyvain unifies the Lloyd algorithm for k-means clustering and the
%       Louvain algorithm for modularity maximization, and thus shows
%       equivalences between modularity maximization, k-means clustering,
%       and spectral clustering.
%
%       The normalized modularity maximization is equivalent to k-means
%       clustering of data after degree correction. When the input is a data
%       rather than a network matrix, degree correction is implemented via
%       an approximately (but not exactly) equivalent process of global-
%       signal regression. Ultimately, degree correction and global-signal
%       regression are both approximately equivalent to the subtraction of
%       the rank-one approximation of the data.
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
    args.acceptance (1, 1) double ...
        {mustBeInRange(args.acceptance, 0, 1)} = 0.5
    args.maxiter (1, 1) {mustBeInteger, mustBePositive} = 1000
    args.start (1, :) double {mustBeInteger, mustBePositive} = 10
    args.verbose (1, 1) logical = false;
end

% Get dimensions
[n, t] = size(X);
assert(all(vecnorm(X, 2, 2) > 0, "all"), "Input data must not contain zero rows or NaN values.")

% Get network matrix
if args.similarity == "network"
    W = X;
    X = [];
else
    W = [];
end
assert(isequal(size(W, 1), size(W, 2)) && all(W - W' < eps("single"), "all"), ...
    "Network matrix must be symmetric or similarity metric must not be set to ""network"".")

% Test non-negativity for spectral and modularity
if ismember(objective, ["modularity" "spectral"])
    err = "Similarity matrix for " + objective + ...
        "clustering must not contain negative values.";
    if args.similarity == "network"
        assert(all(W >= 0, "all"), err);
    elseif (objective == "spectral") && (n < 1e4)
        assert(all(X * X' >= 0, "all"), err)
    elseif (objective == "spectral")
        warning("Not checking similarity matrix for negative values because " + ...
            "number of nodes > 1e4. Ensure that this matrix has no negative " + ...
            "values for compatibility with " + objective + " clustering.")
    end
end

% Process starts
if isscalar(args.start)
    r = args.start;
    assert(k > 0, "Specify number of modules or starting module assignment.")
elseif isvector(args.start)
    r = 1;
    args.start = reshape(args.start, 1, []);
    if k==0
        k = max(args.start);
    end
    assert(length(args.start) == n, "Starting module assignment must have length n.")
    assert(isequal(unique(args.start), 1:k), "Starting module assignments must contain values 1 to k.")
end

assert(k < n, "Number of modules must be smaller than number of nodes.")

if objective == "modularity"
    if args.similarity == "network"
        W = moderemoval(W, "degree");
    else
        X = moderemoval(X, "global");
    end
    objective = "kmeans";
end

% Center data
if ismember(args.similarity, ["cov", "corr"])
    X = X - mean(X, 2);
end

% Normalize data
if ismember(args.similarity, ["cosim", "corr"])
    X = X ./ vecnorm(X, 2, 2);
elseif ismember(args.similarity, ["dot", "cov"])
    X = X / sqrt(t);
end

% Get self-connections
if args.similarity == "network"
    Wii = diag(W)';                         % network self connections
else
    Wii = sum(X.^2, 2)';                    % data sum of squares
end

if isscalar(args.start)
    % Use network as data for kmeans++ initialization
    if args.similarity == "network"
        X = W;
    end
    Lx = vecnorm(X, 2, 2);
end

Q = - inf;
% Run kmeans and keep best output
for i = 1:r
    % kmeans++ initialization
    if isscalar(args.start)
        % select the first seed uniformly at random
        G0 = nan(k, t);
        G0(1, :) = X(randi(n), :);

        % select the other seeds with a probabilistic model
        minDist = inf(n, 1);
        for j = 2:k
            G0j = G0(j-1, :);
            minDist = min(minDist, 1 - X * G0j' ./ (Lx * norm(G0j)));
            sampleProbability = minDist / sum(minDist);
            P = [0 cumsum(sampleProbability)]; P(end) = 1;
            G0(j, :) = X(find(rand < P, 1), :);
        end
        [~, M0] = max(G0' * X, [], 1);         % initialize modules

        % M0 = randi(k, 1, n);                 % initial module partition
        % M0(randperm(n, k)) = 1:k;            % ensure there are k modules
    elseif isvector(args.start)
        M0 = args.start;
    end
    [M1, Q1] = run_loyvain(M0, X, W, Wii, n, k, objective, args, i);
    if mean(Q1) > mean(Q)
        Q = Q1;
        M = M1;
    end
end

end

function [M, Q] = run_loyvain(M, X, W, Wii, n, k, objective, args, replicate_i)

Idx = M + k*(0:n-1);                        % two-dimensional indices of M

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

for v = 1:args.maxiter
    switch objective
        case "kmeans"
            delta_Q = ...
                ((2 * Smn      + Wii) - Cii_nrm    ) ./ (N     + 1) - ...
                ((2 * Smn(Idx) - Wii) - Cii_nrm(M)') ./ (N(M)' - 1);
        case "spectral"
            delta_Q = ...
                ((2 * Smn      + Wii) - Cii_nrm     .* Sn) ./ (Sm     + Sn) - ...
                ((2 * Smn(Idx) - Wii) - Cii_nrm(M)' .* Sn) ./ (Sm(M)' - Sn);
    end
    delta_Q(Idx) = 0;

    % Update if improvements
    [max_delta_Q, M_new] = max(delta_Q);
    if max(max_delta_Q) > 1e-10
        I = find(M ~= M_new);
        n_i = numel(I);
        if args.acceptance < 1
            % Accept a random fraction of swaps
            s_i = max(1, round(args.acceptance * n_i));
            I = I(randperm(n_i, s_i));
            n_i = numel(I);
        end

        while 1
            MMI     = sparse(    M(I), 1:n_i, 1, k, n_i);
            MMI_new = sparse(M_new(I), 1:n_i, 1, k, n_i);

            % Get delta modules and make modules non-empty
            delta_MMI = (MMI_new - MMI);
            N_new = N + sum(delta_MMI, 2);
            ix0 = find(~N_new);
            if isempty(ix0)
                break;
            else
                M_new(I(randperm(n_i, numel(ix0)))) = ix0;
            end
        end

        % Update N, M, MM, and M_Idx
        N = N_new;
        M(I) = M_new(I);
        Idx(I) = M_new(I) + k*(I-1);
        MM = sparse(M, 1:n, 1);

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
    else
        break;
    end
    if args.verbose
        fprintf("Attempt: %5d.  Iteration: %5d.  Swaps: %5d.  Improvement: %5.3f\n", ...
            replicate_i, v, n_i, max(max_delta_Q))
    end
    if v == args.maxiter
        warning("Algorithm did not converge after %d iterations.", v)
    end
end

% Return objective
Q = sum(Cii_nrm);

end
