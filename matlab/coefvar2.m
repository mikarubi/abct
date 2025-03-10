function CV2 = coefvar2(W)
% COEFVAR2 Squared coefficient of variation.
%
%   CV2 = coefvar2(W)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%   Outputs:
%       CV2: Squared coefficient of variation.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
end

CV2 = var(W, 0, 2) ./ mean(W, 2).^2;
