function [CV2, CV2_nrm] = coefvar2(W)
% COEFVAR2 Squared coefficient of variation.
%
%   [CV2, CV2_nrm] = coefvar2(W)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%   Outputs:
%       CV2: Vector of squared coefficients of variation (length n).
%       CV2_nrm: Vector of normalized squared coefficients of variation (length n).
%
%   Methodological notes:
%       The squared coefficient of variation, or CV2, is the ratio of the
%       variance to the square of the mean. CV2 is equivalent to the ratio
%       of the second moment to the square of the first moment.
%
%       The normalized squared coefficient of variation, or CV2_nrm, is
%       defined as (1 - CV2 / (n - 1)) where (n - 1) is the maximum
%       possible value of CV2 for a given mean. CV2_nrm is approximately
%       equivalent to the participation coefficient in homogeneously modular
%       networks, including correlation and common-neighbor networks.
%
%   See also:
%       DEGREES.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal, mustBeNonnegative}
end

n = size(W, 1);
CV2 = var(W, 1, 2) ./ mean(W, 2).^2;
CV2_nrm = 1 - CV2 / (n - 1);
