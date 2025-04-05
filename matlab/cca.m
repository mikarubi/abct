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
%
%       moderm: Mode removal from data.
%           "none": No mode removal (default).
%           "degree": Degree correction.
%           "global": Global signal regression.
%           "rankone": Subtraction of rank-one approximation.
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
%   Methodological note:
%       Weighted canonical correlation or covariance analysis is computed via
%       singular value decomposition of cross-covariance matrix.
%
%       Binary canonical covariance or correlation analysis is computed via
%       Loyvain spectral or k-means co-clustering of cross-covariance matrix.
%       Mode removal converts the k-means co-clustering algorithm into
%       normalized modularity maximization.
%
%   See also:
%       COLOYVAIN, LOYVAIN, MODEREMOVAL.

% Parse inputs and test arguments
arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["canoncorr", "canoncov"])} = "canoncorr"
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary"])} = "weighted"
    moderm (1, 1) string {mustBeMember(moderm, ["none", "degree", "global", "rankone"])} = "none"
end
arguments (Repeating)
    varargin
end
% Do basic checks
[s,  p] = size(X);
[s_, q] = size(Y);
assert(s == s_, "X and Y must have the same number of observations.")
assert(k <= min(p, q), "k must not exceed number of features in X or Y.")

if moderm ~= "none"
    X = moderemoval(X, moderm);
    Y = moderemoval(Y, moderm);
end

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
                [A, R, B] = svds(X' * Y, k);
                R = diag(R);
            case "canoncorr"
                [ux, sx, vx] = svd(X, "econ", "vector");
                rankx = nnz(sx > length(X) * eps(max(sx)));
                if rankx < length(sx)
                    warning("X is not full rank.")
                    sx = sx(1:rankx);
                    ux = ux(:, 1:rankx);
                    vx = vx(:, 1:rankx);
                end

                [uy, sy, vy] = svd(Y, "econ", "vector");
                ranky = nnz(sy > length(Y) * eps(max(sy)));
                if ranky < length(sy)
                    warning("Y is not full rank.")
                    sy = sy(1:ranky);
                    uy = uy(:, 1:ranky);
                    vy = vy(:, 1:ranky);
                end

                [Uw,  R, Vw] = svds(ux' * uy, k);
                A = vx * diag(1./sx) * Uw;
                B = vy * diag(1./sy) * Vw;
                R = diag(R);
        end
    case "binary"
        numbatches = min(32, min(p, q));
        switch type
            case "canoncov"; objective = "kmeans";
            case "canoncorr"; objective = "spectral";
        end
        opts = [{objective}, {"cov"}, {"numbatches"}, {numbatches}, varargin];
        [Mx, My, ~, R] = coloyvain(X', Y', k, opts{:});
        [R, ix] = sort(R, "descend");
        A = zeros(p, k);
        B = zeros(q, k);
        for h = 1:k
            A(Mx == ix(h), h) = 1;
            B(My == ix(h), h) = 1;
        end
end

U = X * A;
V = Y * B;
