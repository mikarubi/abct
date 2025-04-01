function [Mx, My, R] = coloyvain(X, Y, k, objective, varargin)
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
arguments
    X
    Y
    k
    objective
end
arguments (Repeating)
    varargin
end
assert(mod(length(varargin), 2) == 0, "The input must comprise the " + ...
    "required 'X', 'Y', 'k', and 'objective' arguments, followed by the " + ...
    "optional paired Name=[Value] arguments.")

% consolidate arguments
Args = struct(varargin{:}); clear varargin
Args.X = X; clear X;
Args.Y = Y; clear Y;
Args.k = k;
Args.objective = objective; clear objective;
Args = namedargs2cell(Args);

Args = loyv.step0_args("coloyvain", Args{:});   % parse and test arguments
Args = loyv.step1_proc_coloyvain(Args);         % process inputs
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
            Ox = full(sum(MMx0, 2));
            Oy = full(sum(MMy0, 2));
        case "cospectral"
            Ox = Args.Wx;
            Oy = Args.Wy;
    end
    C0_nrm = (MMx0 * Args.Wxy * MMy0') ./ sqrt(Ox .* Oy');

    % align modules
    Mx1 = zeros(size(Mx0));
    My1 = zeros(size(My0));
    for h = 1:k
        [ix, iy] = find(C0_nrm == max(C0_nrm, [], "all"));
        Mx1(Mx0 == ix) = h;
        My1(My0 == iy) = h;
        C0_nrm(ix, :) = nan;
        C0_nrm(:, iy) = nan;
    end

    % fixed point iteration until convergence
    for v = 1:Args.maxiter
        My0 = My1;
        [Mx1,  ~] = loyv.step4_run(Args, Args.Wxy,  Mx1, My1, Args.Wx, Args.Wy);   % optimize Mx
        [My1, R1] = loyv.step4_run(Args, Args.Wxy', My1, Mx1, Args.Wy, Args.Wx);   % optimize My
        if isequal(My0, My1)    % if identical, neither Mx1 nor My1 will change
            break
        end
    end

    % check if replicate has improved on previous result
    if R1 > R
        if ismember(Args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, R1, R1 - R);
        end
        R = R1;
        Mx = Mx1;
        My = My1;
    end
end

end
