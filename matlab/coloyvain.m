function [Mx, My, R, R_all] = coloyvain(varargin)
% COLOYVAIN K-modularity, k-means, or spectral co-clustering
%
%   [Mx, My, R] = coloyvain(W, k)
%   [Mx, My, R] = coloyvain(X, Y, k)
%   [Mx, My, R] = coloyvain(_, objective, similarity)
%   [Mx, My, R] = coloyvain(_, objective, similarity, Name=Value)
%
%   Inputs:
%
%       W: Bipartite network matrix of size p x q.
%
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
%           See LOYVAIN for all options.
%
%       similarity: Type of similarity.
%           See LOYVAIN for all options.
%
%       Name=[Value] Arguments.
%           See LOYVAIN for all Name=Value arguments.
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

%% Parse, process, and test arguments

n_args = length(varargin);
n_args_num = find(cellfun(@(x) ischar(x) || isstring(x), varargin), 1, "first") - 1;
switch n_args_num
    case 2
        [W, k] = deal(varargin{1:n_args_num});
        [X, Y] = deal(0);
    case 3
        [X, Y, k] = deal(varargin{1:n_args_num});
        W = 0;
    otherwise
        error("Wrong number of input arguments.")
end
varargin = varargin(n_args_num+1:end);
if n_args >= n_args_num + 1
    varargin = [varargin(2:end), {"objective"}, varargin(1)];
    if n_args >= n_args_num + 2
        varargin = [varargin(2:end), {"similarity"}, varargin(1)];
    end
end

% Parse arguments
Args = loyv.step0_args("method", "coloyvain", "W", W, "X", X, "Y", Y, "k", k, varargin{:});
clear W X Y k objective similarity

% Process initial arguments
Args = loyv.step1_proc_coloyvain(Args);

% Test arguments
loyv.step2_test(Args.X, Args.W, Args.px, Args.k, Args);
loyv.step2_test(Args.Y, Args.W, Args.py, Args.k, Args);

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
            Ox = ones(Args.px, 1);
            Oy = ones(Args.py, 1);
        case "cospectral"
            Ox = sum(Args.W, 2);
            Oy = sum(Args.W, 1)';
    end
    C0_nrm = (MMx0 * Args.W * MMy0') ./ sqrt((MMx0 * Ox) * (MMy0 * Oy)');

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
        Mx1 = loyv.step4_run(Args, Args.W, Mx1, My1);                   % optimize Mx
        [My1, R1, R1_all] = loyv.step4_run(Args, Args.W', My1, Mx1);    % optimize My
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
