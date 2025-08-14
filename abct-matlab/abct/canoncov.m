function [A, B, U, V, R] = canoncov(X, Y, k, type, corr, resid, varargin)
% CANONCOV Canonical covariance analysis (aka partial least squares)
%          Canonical correlation analysis.
%
%   [A, B, U, V, R] = canoncov(X, Y, k)
%   [A, B, U, V, R] = canoncov(X, Y, k, type)
%   [A, B, U, V, R] = canoncov(X, Y, k, type, corr)
%   [A, B, U, V, R] = canoncov(X, Y, k, type, corr, resid)
%   [A, B, U, V, R] = canoncov(X, Y, k, type, corr, resid, Name=Value)
%
%   Inputs:
%       X: Data matrix of size n x p, where
%          n is the number of data points and
%          p is the number of features.
%
%       Y: Data matrix of size n x q, where
%          n is the number of data points and
%          q is the number of features.
%
%       k: Number of canonical components (positive integer).
%
%       type: Weighted or binary canonical analysis.
%           "weighted": Weighted canonical analysis (default).
%           "binary": Binary canonical analysis.
%
%       corr: Canonical correlation analysis (logical scalar).
%           0: Canonical covariance analysis (default).
%           1: Canonical correlation analysis.
%
%       resid: Global residualization (logical scalar).
%           0: No global residualization (default).
%           1: Global residualization via degree correction.
%
%       Name=[Value] Arguments
%           (binary canonical analysis only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       A: Canonical coefficients of X (size p x k).
%       B: Canonical coefficients of Y (size q x k).
%       U: Canonical components of X (size n x k).
%       V: Canonical components of Y (size n x k).
%       R: Canonical covariances or correlations (size k x k).
%          If type is "weighted", R denotes the actual covariances or
%          correlations. If type is "binary", R denotes the
%          normalized covariances or correlations.
%
%   Methodological notes:
%       Weighted canonical correlation or covariance analysis is computed via
%       singular value decomposition of cross-covariance matrix.
%
%       Binary canonical covariance analysis is computed via co-Loyvain
%       k-means clustering of cross-covariance matrix. This analysis
%       produces binary orthogonal canonical coefficients.
%
%       Binary canonical covariance analysis is computed via co-Loyvain
%       k-means clustering of _whitened_ cross-covariance matrix. This
%       analysis produces binary orthogonal canonical coefficients for 
%       the whitened matrix. However, the output coefficients after
%       dewhitening will, in general, not be binary.
%
%       Global residualization is implemented via generalized degree
%       correction, and converts k-means co-clustering into k-modularity
%       co-maximization.
%
%   See also:
%       COLOYVAIN, LOYVAIN, RESIDUALN.

% Parse inputs and test arguments
arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["weighted", "binary"])} = "weighted"
    corr (1, 1) logical = false
    resid (1, 1) logical = false
end
arguments (Repeating)
    varargin
end

% Basic checks
[n,  p] = size(X);
[n_, q] = size(Y);
assert(n == n_, "X and Y must have the same number of data points.")
assert(k <= min(p, q), "k must not exceed number of features in X or Y.")

% Initial processing
if type == "weighted"
    if ~isempty(varargin)
        warning("Ignoring Name=Value arguments for weighted analysis.")
    end
end

% Global residualization or centering
if resid       % Degree correction automatically centers data
    X = residualn(X, "degree");
    Y = residualn(Y, "degree");
else
    X = X - mean(X, 1);
    Y = Y - mean(Y, 1);
end

% Set up problem
if corr
    [Ux, Sx, Vx] = svdsketch(X);
    [Uy, Sy, Vy] = svdsketch(Y);
    Z = Vx * Ux' * Uy * Vy';
else
    Z = X' * Y;
end

% Solve problem
if type == "weighted"
    [A, R, B] = svds(Z, k);
    R = diag(R);
else
    [Mx, My, ~, R] = coloyvain(Z, k, "kmeans", "network", ...
        "numbatches", min(32, min(p, q)), varargin{:});
    [R, ix] = sort(R, "descend");
    A = zeros(p, k);
    B = zeros(q, k);
    for h = 1:k
        A(Mx == ix(h), h) = 1;
        B(My == ix(h), h) = 1;
    end
end

% Recover coefficients
if corr
    A = Vx / Sx * Vx' * A;
    B = Vy / Sy * Vy' * B;
end

U = X * A;
V = Y * B;
