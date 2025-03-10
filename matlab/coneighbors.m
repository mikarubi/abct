function B = coneighbors(W, p)
% CONEIGHBORS Common-neighbors matrix of network matrix
%
%   B = coneighbors(W, p)
%
%   Inputs:
%       W: Network matrix of size n x n.
%       p: Neighbor fraction
%           Neighbors are the top-p fraction of connections
%           0 < p < 1 (default is 0.1).
%
%   Outputs:
%       B: Integer-valued common-neighbors matrix.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    p (1, 1) double {mustBeInRange(p, 0, 1)} = 0.1
end

A = W > quantile(W, (1-p), 2);
B = A * A';
