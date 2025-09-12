function D = dispersion(W, type, M)
% DISPERSION Dispersion of network matrix
%
%   D = dispersion(W)
%   D = dispersion(W, type, M)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       type: Dispersion type
%           "coefvar2": Squared coefficient of variation (default).
%           "kpartcoef": k-Participation coefficient.
%
%       M: Module vector of length n (if type is "kpartcoef" only).
%
%   Outputs:
%       D: Dispersion vector (length n).
%
%   Methodological notes:
%       The squared coefficient of variation, or CV2, is the ratio of the
%       variance to the square of the mean. CV2 is equivalent to the ratio
%       of the second moment to the square of the first moment.
%
%       The participation coefficient is a popular module-based measure of
%       connectional diversity. The k-participation coefficient is the
%       participation coefficient normalized by module size.
%
%       CV2 is approximately equivalent to the k-participation coefficient
%       in homogeneously modular networks, such as correlation or
%       co-neighbor networks.
%
%   See also:
%       DEGREE.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal, mustBeNonnegative}
    type (1, 1) string {mustBeMember(type, ["coefvar2", "kpartcoef"])} = "coefvar2"
    M (:, 1) double {mustBeInteger, mustBePositive} = []
end

switch type
    case "coefvar2"
        D = var(W, 1, 2) ./ mean(W, 2).^2;
    case "kpartcoef"
        [n, n_] = size(W);
        n__ = length(M);
        assert(n == n_ && n == n__, "W must be a square matrix and M must have the same length as W.")
        k = max(M);
        MM = sparse(1:n, M, 1, n, k);
        kSnm = (W * MM) ./ sum(MM, 1);
        P = kSnm ./ sum(kSnm, 2);
        D = 1 - sum(P.^2, 2);
end
