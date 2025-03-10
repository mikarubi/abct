function [A, B, U, V] = canoncov(X, Y, k, args)
% CANONCOV Canonical covariance analysis (aka Partial least squares)
%
%   [A, B, U, V] = canoncov(X, Y, k)
%   [A, B, U, V] = canoncov(X, Y, k, Name=Value)
%
%   Inputs:
%       X: Input matrix of size n x q.
%       Y: Input matrix of size n x r.
%       k: Number of canonical components.
%
%       Name=[Value] Arguments (Optional):
%
%           Type=[Type of CCA].
%               "standard": Standard CCA (default).
%               "binary": Binary CCA.
%
%   Outputs:
%       A: Canonical coefficients of X.
%       B: Canonical coefficients of Y.
%       U: Canonical scores of X.
%       V: Canonical scores of Y.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    Y (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    args.Type (1, 1) string {mustBeMember(args.Type, ["standard", "binary"])} = "standard"
end

[n, q] = size(X);
[n_, r] = size(Y);
assert(n == n_, "The input matrices must have the same number of rows.")

% Mean center columns instead of rows
X = X - mean(X, 1);
Y = Y - mean(Y, 1);
Z = Y' * X / n;

% Check if k is too large
k = min(k, rank(Z));

switch args.Type
    case "standard"
        % Standard PLS
        [A, ~, B] = svds(Z, k);

    case "binary"
        % Binary PLS
        Ma = loyvain(Z, k, similarity="dot", objective="kmeans");
        Mb = loyvain(Z', k, similarity="dot", objective="kmeans");
        A = full(sparse(1:q, Ma, 1));
        B = full(sparse(1:r, Mb, 1));

        % Choose k largest components
        C = (X * A)' * (Y * B) / n;
        [~, Idx] = sort(C(:), "descend");
        Idx = Idx(1:k);
        I = ceil(Idx / q);
        J = mod(Idx, q);
        [i, j] = ind2sub(size(C), Idx);
        assert(isequal([I J], [i, j]))

        % Find the first non-unique element of I
        [~, uIdx_I] = unique(I, "first", "stable");
        [~, uIdx_J] = unique(J, "first", "stable");
        max_kI = find(diff(uIdx_I) > 1, 1, "first");
        max_kJ = find(diff(uIdx_J) > 1, 1, "first");
        maxk = min([max_kI, max_kJ, k]);
        if maxk < k
            warning("Only %d components supported.", maxk)
        end
        A = A(i(1:maxk), :);
        B = B(j(1:maxk), :);
end

U = X * A;
V = Y * B;

end
