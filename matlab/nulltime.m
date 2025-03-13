function [X0, G0] = nulltime(X, M, s)
% NULLTIME Null timeseries with preserved node-to-module correlation structure.
%
%   [X0, G0] = nulltime(X, M, s)
%
%   This function takes an empirical timeseries matrix and a corresponding
%   module assignment vector. It then generates the following null data:
%
%   1. A set of k modes that have the same cross-node correlation structure
%      as the centroids of k empirical modules.
%
%   2. Null timeseries where the correlation between node i and mode u matches
%      the corresponding empirical correlation between node i and mode u.
%
%   In practice, this procedure generates null timeseries with similar
%   node-to-module correlation structure as the empirical data but are
%   otherwise maximally random.
%
%   Inputs:
%       X: Timeseries matrix (size n x t).
%
%       M: Module assignment vector (length n).
%
%       s: Number of samples for null data (default is 1).
%
%   Outputs:
%       X0: Samples of null timeseries (size n x t x s).
%
%       G0: Samples of null modes (size n x k x s).
%
%   Methodological notes:
%       This function uses nullspace sampling to generate null timeseries.
%       It is memory intensive and may not scale well to large datasets.

arguments
    X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    M (1, :) double {mustBeInteger, mustBePositive}
    s (1, 1) double {mustBeInteger, mustBePositive} = 1
end

k = max(M);
[n, t] = size(X);
assert(length(M) == n, "Module assignment vector must have length n.")

% timeseries mean and variance
MeanX = mean(X, 2);
VarX = var(X, 0, 2);

% normalized centroids
MM = sparse(M, 1:n, 1);
G = normalize(MM * X, 2);

% correlation constraints
Smm = G * G' / (t - 1);     % preserve covnode via rotated eigen-nullspace
Snm = X * G';               % preserve covnodemode via standard nullspace

% generate null timeseries
G0 = zeros(k, t, s);
X0 = zeros(n, t, s);
for i = 1:s
    G0(:,:,i) = covnode_nullspace(Smm, t);
    X0(:,:,i) = covnodemode_nullspace(Snm, G0(:,:,i), MeanX, VarX);
end

end

function X = covnode_nullspace(C, t)
% Inputs
%   C,  correlation matrix
%   t,  number of timepoints
% Outputs
%   X,  timeseries with preserved correlation matrix

% target eigendecomposition
[V, D] = eig(C, "vector");

n = size(C, 1);
X0 = zeros(n, t);
for i = 1:n
    % set up system
    A = [X0(1:i-1,:); ones(1, t)/t];
    b = zeros(1, i);
    Z = null(A);

    % sample from nullspace
    X0(i, :) = nullspace(Z, A, b, D(i), t);
end
X = V * X0;

end

function X = covnodemode_nullspace(Cnm, Xm, MeanX, VarX)
% Inputs
%   Y,  (modes x timepoints) input mode timeseries
%   C,  (nodes x timepoints) input node-mode cov
% Outputs
%   X,   timeseries with preserved node-to-module sum of correlations

t = size(Xm, 2);

% set up system
A = [Xm; ones(1, t)/t];
b = [Cnm MeanX];
Z = null(A);

% sample from nullspace
X = nullspace(Z, A, b, VarX, t);

end

function X = nullspace(Z, A, b, v, t)
% Low-level function for nullspace sampling
%
% Inputs:
%   Z: nullspace
%   A: constraints
%   b: empirical values
%   v: timeseries variance
%   t: timeseries length
%
% Outputs:
%   X: null timeseries

% nullspace dimension
n = size(b, 1);
z = size(Z, 2);

% minimum norm solution
xn = lsqminnorm(A, b')';

% projection radius
radi = (t-1) * v + t * mean(xn, 2).^2 - sum(xn.^2, 2);
assert(all(radi > -1e-10))
radi = sqrt(max(radi, 0));

% a random point on mz-dimensional sphere
q = normalize(randn([z, n]), 1, 'norm', 2);
X = xn + radi .* (Z * q)';

end
