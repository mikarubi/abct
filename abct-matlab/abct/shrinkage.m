function X = shrinkage(X)
% SHRINKAGE Network shrinkage
%
%   X1 = shrinkage(X)
%
%   Inputs:
%       X:  Network matrix of size n x n
%
%   Outputs:
%       X1: Shrunken network matrix.
%
%   Methodological notes:
%       The shrinkage algorithm uses cubic interpolation to "despike" an
%       initial peak in the eigenspectrum.
%
%   See also:
%       RESIDUALN.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
end

assert(isequal(size(X, 1), size(X, 2)) && all(abs(X - X') < eps("single"), "all"), ...
    "Invalid shrinkage: Network matrix must be symmetric.")
[V, D] = eig(X, "vector");
[~, ix] = sort(D, "descend");
V = V(:, ix);
D = D(ix);

n = length(D);
[bk, r0] = polyfit((1:n)', D, 3);
rms0 = r0.normr / sqrt(n);    % get rms
x = rescale(1:n)';
y(n) = 0;
for k = 1:n
    b = bk;
    [bk, rk] = polyfit((k:n)', D(k:n), 3);
    assert(numel((k:n))==(n-k+1))
    rmsk = rk.normr / sqrt(n-k+1);
    y(k) = (rms0 - rmsk) / rms0;
    % detect knee of optimal fit and break
    if k > 1 && ((y(k) - x(k)) < (y(k-1) - x(k-1)))
        break
    end
end

D = polyval(b, (1:n)');
X = V * diag(D) * V';
