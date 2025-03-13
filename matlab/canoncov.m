function [A, B, U, V] = canoncov(X, Y, k, type, varargin)
% CANONCOV Canonical covariance analysis (aka Partial least squares)
%
%   [A, B, U, V] = canoncov(X, Y, k)
%   [A, B, U, V] = canoncov(X, Y, k, type)
%   [A, B, U, V] = canoncov(X, Y, k, type, Name=Value)
%
%   Inputs:
%       X: Input matrix of size n x q.
%          n is the number of observations and q is the number of variables.
%
%       Y: Input matrix of size n x r.
%          n is the number of observations and r is the number of variables.
%
%       k: Number of canonical components.
%
%       type: Type of analysis.
%           "weighted": Standard canonical covariance (default).
%           "binary": Binary canonical covariance.
%
%       Name=[Value] Arguments:
%           Name-value arguments for the Loyvain algorithm.
%           Only used if type = "binary". See LOYVAIN for details.
%
%   Outputs:
%       A: Canonical coefficients of X (size q x k).
%
%       B: Canonical coefficients of Y (size r x k).
%
%       U: Canonical components of X (size n x k).
%
%       V: Canonical components of Y (size n x k).
%
%   Methodological notes:
%       Weighted canonical covariance analysis (aka partial least squares)
%       is computed via the singular value decomposition of the
%       cross-covariance matrix. Binary canonical covariance analysis is
%       computed via Loyvain clustering of the cross-covariance matrix,
%       followed by iterative cluster matching. For convenience the output
%       canonical coefficients A and B are rescaled to have norm 1.
%
%   See also:
%       CANONCORR, LOYVAIN.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["weighted", "binary"])} = "weighted"
end
arguments (Repeating)
    varargin
end
if type == "weighted" && ~isempty(varargin)
    warning("Ignoring Name=Value arguments for weighted gradients.")
end

[n, q] = size(X);
[n_, r] = size(Y);
assert(n == n_, "The input matrices must have the same number of rows.")
assert(k <= min(q, r), "k must not exceed the number of columns in X or Y.")

% Mean center columns instead of rows
X = X - mean(X, 1);
Y = Y - mean(Y, 1);
Z = X' * Y / n;

switch type
    case "weighted"
        % Standard PLS
        [A, ~, B] = svds(Z, k);

    case "binary"
        % Binary PLS
        Ma = loyvain(Z, k, "kmeans", varargin{:}, similarity="dot");
        Mb = loyvain(Z', k, "kmeans", varargin{:}, similarity="dot");
        A = full(sparse(1:q, Ma, 1));
        B = full(sparse(1:r, Mb, 1));
        A = A ./ vecnorm(A, 2, 1);
        B = B ./ vecnorm(B, 2, 1);

        % Choose k largest components
        C = (X * A)' * (Y * B) / n;
        [~, Idx] = sort(C(:), "descend");
        [I, J] = ind2sub([k, k], Idx(1:k));

        % Find the first non-unique element of I
        [~, Idx_uI] = unique(I, "first");
        [~, Idx_uJ] = unique(J, "first");
        max_kI = find(diff([0; sort(Idx_uI)]) == 1, 1, "last");
        max_kJ = find(diff([0; sort(Idx_uJ)]) == 1, 1, "last");
        maxk = min([max_kI, max_kJ, k]);
        if maxk < k
            warning("Only %d components supported.", maxk)
        end
        A = A(:, I(1:maxk));
        B = B(:, J(1:maxk));
end

U = X * A;
V = Y * B;

end
