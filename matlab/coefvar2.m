function CV2 = coefvar2(W)
% COEFVAR2 Squared coefficient of variation.
%
%   CV2 = coefvar2(W)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%   Outputs:
%       CV2: Squared coefficient of variation vector of size n.
%
%   Methodological notes:
%       The squared coefficient of variation is the ratio of the variance to
%       the square of the mean. It is exactly equivalent to the ratio of the
%       second degree to the square of the first degree. It is approximately
%       equivalent to the participation coefficient in homogeneously modular
%       networks, including correlation and common-neighbor networks.
%
%   See also:
%       DEGREES, CONEIGHBORS.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
end

CV2 = var(W, 0, 2) ./ mean(W, 2).^2;
