function [A, B, R, U, V] = cca(X, Y, k, type, weight, moderm, varargin)
% CCA Canonical correlation or covariance analysis
%
%   [A, B, R, U, V] = cca(X, Y, k)
%   [A, B, R, U, V] = cca(X, Y, k, type)
%   [A, B, R, U, V] = cca(X, Y, k, type, weight)
%   [A, B, R, U, V] = cca(X, Y, k, type, weight, moderm)
%   [A, B, R, U, V] = cca(X, Y, k, type, weight, moderm, Name=Value)
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
%       k: Number of canonical components (positive integer).
%
%       type: Type of canonical analysis.
%           "canoncov": Canonical covariance analysis,
%                       aka partial least squares (default).
%           "canoncorr": Canonical correlation analysis.
%
%       weight: Weighted or binary canonical analysis.
%           "weighted": Weighted canonical analysis (default).
%           "binary": Binary canonical analysis.
%           "hybrid": Hybrid canonical analysis 
%                     (only compatible with canonical correlation).
%
%       moderm: First-mode removal (logical scalar).
%           0: No first-mode removal (default).
%           1: First-mode removal via degree correction.
%
%       Name=[Value] Arguments (binary canonical analysis only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       A: Canonical coefficients of X (size p x k).
%       B: Canonical coefficients of Y (size q x k).
%       R: Canonical covariances or correlations (size k x 1)
%       U: Canonical components of X (size s x k).
%       V: Canonical components of Y (size s x k).
%
%   Methodological notes:
%       Weighted canonical correlation or covariance analysis is computed via
%       singular value decomposition of cross-covariance matrix.
%
%       Binary canonical covariance (respectively canonical correlation)
%       analysis is computed via Loyvain k-means (respectively Loyvain spectral)
%       co-clustering of cross-covariance matrix. This analysis produces binary
%       orthogonal canonical coefficients.
%
%       Hybrid canonical correlation analysis is computed via Loyvain k-means
%       co-clustering of whitened cross-covariance matrix. This analysis produces
%       orthogonal canonical components, but not binary canonical coefficients.
%
%       First-mode removal is performed via generalized degree correction, and
%       converts k-means co-clustering into normalized modularity maximization.
%
%   See also:
%       COLOYVAIN, LOYVAIN, MODEREMOVAL.

% Parse inputs and test arguments
arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["canoncorr", "canoncov"])} = "canoncov"
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary", "hybrid"])} = "weighted"
    moderm (1, 1) logical = false
end
arguments (Repeating)
    varargin
end

% Basic checks
[s,  p] = size(X);
[s_, q] = size(Y);
assert(s == s_, "X and Y must have the same number of observations.")
assert(k <= min(p, q), "k must not exceed number of features in X or Y.")
assert(weight ~= "hybrid" || type == "canoncorr", ...
    "Hybrid analysis is only compatible with canonical correlation.")

% Initial processing
if weight == "weighted"
    if ~isempty(varargin)
        warning("Ignoring Name=Value arguments for weighted analysis.")
    end
elseif (weight == "hybrid") || (type == "canoncov")
    objective = "kmeans";
else
    objective = "spectral";
end

% First-mode removal or centering
if moderm       % Degree correction automatically centers data
    X = moderemoval(X, "degree");
    Y = moderemoval(Y, "degree");
else
    X = X - mean(X, 1);
    Y = Y - mean(Y, 1);
end

% Set up problem
if (type == "canoncorr") && (weight ~= "binary")
    [Ux, Sx, Vx] = svd(X, "econ", "vector");
    rankx = nnz(Sx > length(X) * eps(max(Sx)));
    if rankx < length(Sx)
        warning("X is not full rank.")
        Sx = Sx(1:rankx);
        Ux = Ux(:, 1:rankx);
        Vx = Vx(:, 1:rankx);
    end

    [Uy, Sy, Vy] = svd(Y, "econ", "vector");
    ranky = nnz(Sy > length(Y) * eps(max(Sy)));
    if ranky < length(Sy)
        warning("Y is not full rank.")
        Sy = Sy(1:ranky);
        Uy = Uy(:, 1:ranky);
        Vy = Vy(:, 1:ranky);
    end
else
    Ux = X;
    Uy = Y;
end

% Solve problem
if weight == "weighted"
    [A, R, B] = svds(Ux' * Uy, k);
    R = diag(R);
else
    numbatches = min(32, min(p, q));
    opts = [{objective}, {"dot"}, {"numbatches"}, {numbatches}, varargin];
    [Mx, My, ~, R] = coloyvain(Ux', Uy', k, opts{:});
    [R, ix] = sort(R, "descend");
    A = zeros(p, k);
    B = zeros(q, k);
    for h = 1:k
        A(Mx == ix(h), h) = 1;
        B(My == ix(h), h) = 1;
    end
end

% Recover coefficients
if (type == "canoncorr") && (weight ~= "binary")
    A = Vx * diag(1./Sx) * A;
    B = Vy * diag(1./Sy) * B;
end

U = X * A;
V = Y * B;
