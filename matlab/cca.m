function [A, B, U, V] = cca(X, Y, k, type, weight, varargin)
% CCA Canonical correlation or covariance analysis
%
%   [A, B, U, V] = cca(X, Y, k)
%   [A, B, U, V] = cca(X, Y, k, type)
%   [A, B, U, V] = cca(X, Y, k, type, weight, Name=Value)
%
%   Inputs:
%       X: Data matrix of size s x p, where
%          s is the number of observations and
%          p is the number of features of X
%
%       Y: Data matrix of size s x q, where
%          s is the number of observations and
%          q is the number of features of Y
%
%       k: Number of canonical components (positive integer).
%
%       type: Type of canonical analysis.
%           "canoncorr": Canonical correlation analysis (default).
%           "canoncov": Canonical covariance analysis(aka partial least squares).
%
%       weight: Weighted or binary canonical analysis.
%           "weighted": Weighted canonical analysis (default).
%           "binary": Binary canonical analysis.
%
%       Name=[Value] Arguments (binary canonical analysis only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       A: Canonical coefficients of X (size p x k).
%       B: Canonical coefficients of Y (size q x k).
%       U: Canonical components of X (size s x k).
%       V: Canonical components of Y (size s x k).
%
%   Methodological note:
%       Binary canonical correlation or covariance analysis is computed with
%       spectral or k-means co-Loyvain clustering of cross-covariance matrix.
%
%   See also:
%       CANONCORR, LOYVAIN.

% Parse inputs and test arguments
arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["canoncorr", "canoncov"])} = "canoncorr"
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary"])} = "weighted"
end
arguments (Repeating)
    varargin
end
% Do basic checks
[s,  p] = size(X);
[s_, q] = size(Y);
assert(s == s_, "X and Y must have the same number of observations.")
assert(k <= min(p, q), "k must not exceed number of features in X or Y.")

switch weight
    case "weighted"
        if ~isempty(varargin)
            warning("Ignoring Name=Value arguments for weighted analysis.")
        end

        % Center data
        X = X - mean(X, 1);
        Y = Y - mean(Y, 1);
        switch type
            case "canoncov"
                [A, ~, B] = svds(X' * Y, k);
            case "canoncorr"
                [ux, sx, vx] = svd(X, "econ");
                [uy, sy, vy] = svd(Y, "econ");
                [Uw,  ~, Vw] = svds(ux' * uy, k);
                A = vx * sx^(-1) * Uw;
                B = vy * sy^(-1) * Vw;
        end 
    case "binary"
        switch type
            case "canoncov"
                [Mx, My] = coloyvain(X', Y', k, "kmeans", "cov");
            case "canoncorr"
                [Mx, My] = coloyvain(X', Y', k, "spectral", "cov");
        end
        A = full(sparse(1:p, Mx, 1));
        B = full(sparse(1:q, My, 1));
end

U = X * A;
V = Y * B;
