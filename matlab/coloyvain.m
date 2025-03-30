function [Mx, My, r] = coloyvain(X, Y, k, objective, varargin)
% COLOYVAIN Normalized modularity, k-means, or spectral co-clustering
%
%   [Mx, My, R] = coloyvain(X, Y, k)
%   [Mx, My, R] = coloyvain(X, Y, k, objective)
%   [Mx, My, R] = coloyvain(X, Y, k, objective, Name=Value)
%
%   Inputs:
%       X: Data matrix of size p x s, where
%          p is the number of features of X and
%          s is the number of observations
%
%       Y: Data matrix of size q x s, where
%          q is the number of features of Y and
%          s is the number of observations
%
%       k: Number of modules (positive integer or 0).
%           Set to 0 to infer number from initial module assignment.
%
%       objective: Co-clustering objective.
%           "modularity": Normalized modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Spectral clustering objective (normalized cut).
%
%       Name=[Value] Arguments:
%
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       Mx: Vector of module assignments for X (length p).
%       My: Vector of module assignments for Y (length q).
%
%   Methodological notes:
%       Binary canonical analysis is computed via Loyvain co-clustering of the
%       cross-correlation or cross-covariance matrix.
%
%   See also:
%       CANONCORR, LOYVAIN.

% consolidate arguments
Args = struct(varargin{:}); clear varargin
Args.X = X; clear X;
Args.Y = Y; clear Y;
Args.k = k;
Args.objective = objective; clear objective;
Args = namedargs2cell(Args);

Args = loyv.step0_args("coloyvain", Args{:});   % parse and test arguments
Args = loyv.step1_proc_coloyvain(Args);         % process inputs
loyv.step2_test(Args.X, Args.p, Args.W, Args.k, Args);
loyv.step2_test(Args.Y, Args.q, Args.W, Args.k, Args);

%% Run algorithm

r = - inf;
for i = 1:Args.replicates

    % initialize
    Mx0 = loyv.step3_init([], Args.p, Args.DistX, [], Args);
    My0 = loyv.step3_init([], Args.q, Args.DistY, [], Args);

    % get between-module correlations
    MMx0 = sparse(1:p, Mx0, 1);
    MMy0 = sparse(1:q, My0, 1);
    switch Args.objective
        case "kmeans"
            Ox = eye(Args.p);
            Oy = eye(Args.q);
        case "spectral"
            Ox = Args.Cx;
            Oy = Args.Cy;
    end
    R0 = MMx0' * Args.W * MMy0 ./ sqrt((MMx0' * Ox * MMx0) .* (MMy0' * Oy * MMy0));

    % align modules
    for h = 1:k
        [ix, iy] = find(R0 == max(R0, [], "all"));
        Mx0(Mx0 == ix) = - h;
        My0(My0 == iy) = - h;
        R0(ix, :) = nan;
        R0(:, iy) = nan;
    end
    Mx0 = - Mx0;
    My0 = - My0;

    [Mx1, r0] = loyv.step4_run(Args, Mx0, i);   % run
    [My1, r1] = loyv.step4_run(Args, My0, i);   % run
    assert(r1 >= r0)
    if r1 > r
        if ismember(Args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, r1, r1 - r);
        end
        r = r1;
        Mx = Mx1;
        My = My1;
    end
end

end