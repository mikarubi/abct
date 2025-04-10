function [Mx, My, R, R_all] = coloyvain(X, Y, k, objective, similarity, varargin)
% COLOYVAIN K-modularity, k-means, or spectral co-clustering
%
%   [Mx, My, R] = coloyvain(X, Y, k)
%   [Mx, My, R] = coloyvain(X, Y, k, objective, similarity)
%   [Mx, My, R] = coloyvain(X, Y, k, objective, similarity, Name=Value)
%
%   Inputs:
%       X: Data matrix of size s x p, where
%          s is the number of data points and
%          p is the number of features.
%
%       Y: Data matrix of size s x q, where
%          s is the number of data points and
%          q is the number of features.
%
%       k: Number of modules (positive integer).
%
%       objective: Clustering objective.
%           "kmodularity": K-modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Modified spectral clustering objective.
%
%       similarity: Type of similarity.
%           "corr": Pearson correlation coefficient (default).
%               A magnitude-normalized dot product of mean-centered vectors.
%           "cosim": Cosine similarity.
%               A normalized dot product.
%           "cov":  Covariance.
%               A dot product of mean-centered vectors.
%           "dot": Dot product.
%               A sum of an elementwise vector product.
%
%       Name=[Value] Arguments:
%
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       Mx: Vector of module assignments for X (length p).
%       My: Vector of module assignments for Y (length q).
%       R: Value of maximized objective.
%
%   Methodological notes:
%       Coloyvain simultaneously clusters X and Y via Loyvain
%       co-clustering of the cross-similarity matrix.
%
%   See also:
%       LOYVAIN, CCA.

arguments
    X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    Y (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    k (1, 1) double {mustBeInteger, mustBePositive}
    objective (1, 1) string {mustBeMember(objective, ...
        ["kmodularity", "kmeans", "spectral"])} = "kmodularity"
    similarity (1, 1) string {mustBeMember(similarity, ...
        ["corr", "cosim", "cov", "dot"])} = "corr"
end
arguments (Repeating)
    varargin
end

% parse, process, and test arguments
Args = loyv.step0_args("method", "coloyvain", "X", X, "Y", Y, "k", k, ...
    "objective", objective, "similarity", similarity, varargin{:});
clear X Y k objective similarity
Args = loyv.step1_proc_coloyvain(Args);
loyv.step2_test(Args.X, Args.Wxy, Args.px, Args.k, Args);
loyv.step2_test(Args.Y, Args.Wxy, Args.py, Args.k, Args);

%% Run algorithm

R = - inf;
for i = 1:Args.replicates
    Args.replicate_i = i;

    % initialize
    Mx0 = loyv.step3_init([], [], Args.DistX, Args.px, Args);
    My0 = loyv.step3_init([], [], Args.DistY, Args.py, Args);

    % get between-module correlations
    MMx0 = sparse(Mx0, 1:Args.px, 1);
    MMy0 = sparse(My0, 1:Args.py, 1);
    switch Args.objective
        case "cokmeans"
            Ox = eye(Args.px);
            Oy = eye(Args.py);
        case "cospectral"
            Ox = Args.Wx;
            Oy = Args.Wy;
    end
    C0_nrm = (MMx0 * Args.Wxy * MMy0') ./ ...
        sqrt(diag(MMx0 * Ox * MMx0') * diag(MMy0 * Oy * MMy0')');

    % align modules
    Mx1 = zeros(size(Mx0));
    My1 = zeros(size(My0));
    for h = 1:Args.k
        [ix, iy] = find(C0_nrm == max(C0_nrm, [], "all"), 1);
        Mx1(Mx0 == ix) = h;
        My1(My0 == iy) = h;
        C0_nrm(ix, :) = nan;
        C0_nrm(:, iy) = nan;
    end

    % fixed point iteration until convergence
    for v = 1:Args.maxiter
        My0 = My1;
        Mx1 = loyv.step4_run(Args, Args.Wxy,  Mx1, My1, Args.Wx, Args.Wy, Args.Wx_ii);   % optimize Mx
        [My1, R1, R1_all] = loyv.step4_run(Args, Args.Wxy', My1, Mx1, Args.Wy, Args.Wx, Args.Wy_ii);   % optimize My
        if isequal(My0, My1)    % if identical, neither Mx1 nor My1 will change
            break
        end
        if Args.display == "iteration"
            fprintf("Replicate: %4d.    Iteration: %4d.    Objective: %4.4f.\n", ...
                Args.replicate_i, v, R1)
        end
        if v == Args.maxiter
            warning("Algorithm did not converge after %d iterations.", v)
        end
    end

    % check if replicate has improved on previous result
    if (R1 - R) > Args.tolerance            % test for increase
        if ismember(Args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, R1, R1 - R);
        end
        R = R1;
        Mx = Mx1;
        My = My1;
        R_all = R1_all;
    end
end

end
