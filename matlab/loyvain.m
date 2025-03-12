function [M, Q] = loyvain(X, k, objective, args)
% LOYVAIN Normalized modularity, k-means, or spectral clustering
%
%   [M, Q] = loyvain(X, k, objective, Name=Value)
%
%   Inputs:
%       X: Network matrix of size n x n, or data matrix of size n x t.
%          n is the number of nodes and t is the number of observations.
%
%       k: Number of clusters (positive integer).
%
%       objective: Clustering objective.
%           "modularity": Normalized modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Spectral clustering objective (normalized cut).
%
%       Name=[Value] Arguments (Optional):
%
%           Similarity=[Type of similarity].
%               "precomputed": Input similarity network (default)
%                   The first input is a symmetric similarity matrix.
%               "corr": Pearson correlation coefficient.
%                   A common measure of linear association, defined as a
%                   normalized dot product of mean-centered vectors.
%               "cosim": Cosine similarity.
%                   A normalized dot product.
%               "cov":  Covariance.
%                   A dot product of mean-centered vectors.
%               "dot": Dot product.
%                   A sum of an elementwise vector product.
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
%                   Number of starts from random initial module assignments.
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
%       Normalized degree-corrected modularity is approximately equivalent
%       to k-means objective after global signal regression (removal of the
%       mean signal from the data). When the input is a data matrix rather
%       than a similarity matrix, degree correction is replaced with global
%       signal regression, which may give slightly different results.
%
%   See also:
%       GRADIENTS, MODEREMOVAL.

arguments
    X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    k (1, 1) double {mustBeInteger, mustBePositive}
    objective (1, 1) string {mustBeMember(objective, ...
        ["kmeans", "spectral", "modularity"])} = "modularity"
    args.similarity (1, 1) string {mustBeMember(args.similarity, ...
        ["dot", "cov", "cosim", "corr", "precomputed"])} = "precomputed"
    args.acceptance (1, 1) double ...
        {mustBeInRange(args.acceptance, 0, 1)} = 0.5
    args.maxiter (1, 1) {mustBeInteger, mustBePositive} = 1000
    args.start (1, :) double {mustBeInteger, mustBePositive} = 10
    args.verbose (1, 1) logical = false;
end

% Get dimensions
[n, t] = size(X);
assert(k < n, "Number of clusters must be smaller than number of nodes.")

% Check precomputed matrix
if args.similarity == "precomputed"
    W = X;
else
    W = [];
end
assert(isequal(size(W, 1), size(W, 2)) && all(W - W' < eps("single"), "all"), ...
    "Ensure similarity matrix is symmetric or change similarity metric.")

% Test non-negativity for spectral and modularity
if ismember(objective, ["modularity" "spectral"])
    err = "Similarity matrix for " + objective + ...
        "clustering must not contain negative values.";
    if args.similarity == "precomputed"
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
elseif isvector(args.start)
    r = 1;
    args.start = reshape(args.start, 1, []);
    assert(length(args.start) == n, "Starting module assignment must have length n.")
end

if objective == "modularity"
    if args.similarity == "precomputed"
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
if args.similarity == "precomputed"
    Wii = diag(W)';                         % network self connections
else
    Wii = sum(X.^2, 2)';                    % data sum of squares
end

Q = - inf;
for i = 1:r
    % Run kmeans and keep best output
    if isscalar(args.start)
        M = randi(k, 1, n);                 % initial module partition
        M(randperm(n, k)) = 1:k;            % ensure there are k modules
    else
        M = args.start;
        assert(isequal(unique(M), 1:k), "Starting module assignments must contain values 1 to k.")
    end
    [M1, Q1] = run_loyvain(M, X, W, Wii, n, k, objective, args, i);
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
if args.similarity == "precomputed"
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
        if args.similarity == "precomputed"
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
        fprintf("Replicate: %d.  Iteration: %d.  Swaps: %d.  Increase: %.3f\n", ...
            replicate_i, v, n_i, max(max_delta_Q))
    end
    if v == args.maxiter
        warning("Algorithm did not converge after %d iterations.", v)
    end
end

% Return objective
Q = sum(Cii_nrm);

end
