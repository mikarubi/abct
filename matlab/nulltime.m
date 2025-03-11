function X0 = nulltime(X, M, t, s)
% NULLTIME Null timeseries with preserved covariance structure.
%
%   This function generates null timeseries with preserved module-to-module
%   covariance structure as well as node-to-module covariance structure.
%
%   X0 = nulltime(X, M, t, s)
%
%   Inputs
%       X,  Timeseries matrix of size n x t.
%       M,  Module assignment vector of length n.
%       t,  Number of timepoints for null data.
%       s,  Number of samples for null data.
%
%   Outputs
%       X0,  Samples of timeseries matrix of size n x t.
%
%   Methodological notes:
%       This function uses nullspace sampling to generate synthetic timeseries.
%       It is memory intensive and may thus not scale well to large datasets.
%
%   See also:
%       NULLSPACE.

n = size(X, 1);

MM = sparse(M, 1:n, 1);                     % two-dimensional representation
G = MM * X;                                 % cluster centroid
G = G - mean(G, 2);
G = G ./ vecnorm(G, 2, 2);

Smm = G * G';
Snm = X * G';                               % dot of centroid with node
X0 = cell(s, 1);
for i = 1:s
    % loop over samples and get null models
    G0 = covnode_nullspace(Smm, t);
    X0{i} = covnodemode_nullspace(Snm, G0);
end

end

function X = covnode_nullspace(C, t)
% Inputs
%   C,  correlation matrix
%   t,  number of timepoints
% Outputs
%   X,  timeseries with preserved correlation matrix

% singular value decomposition
[U, S] = svd(C);

n = size(C, 1);
X0 = zeros(n, t);
for i = 1:n
    % set up system
    A = [X0(1:i-1,:); ones(1, t)/t];
    b = zeros(1, i);
    Z = null(A);

    % sample from nullspace
    X0(i, :) = nullspace(Z, A, b, S(i, i), t);
end
X = U * X0;

end

function X = covnodemode_nullspace(Cnm, Xm)
% Inputs
%   Y,  (modes x timepoints) input mode timeseries
%   C,  (nodes x timepoints) input node-mode cov
% Outputs
%   X,   timeseries with preserved node-to-module sum of correlations

% set up system
A = Xm;
b = Cnm;
Z = null(A);
t = size(Xm, 2);

% sample from nullspace
X = nullspace(Z, A, b, 1, t);

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
