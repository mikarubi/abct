function B = coneighbors(W, p)
% CONEIGHBORS Common-neighbors matrix of network matrix
%
%   B = coneighbors(W, p)
%
%   Inputs:
%       W: Network matrix of size n x n.
%       p: Fraction to define neighbors
%           (Neighbors are the top-p fraction of connections)
%           0 < p < 1 (default is 0.1).
%
%   Outputs:
%       B: Common-neighbors matrix of size n x n.
%
%   Methodological notes:
%       Integer-valued common-neighbors matrices denote the number of common
%       top-p neighbors between all pairs of nodes. These matrices are
%       approximately equivalent to the kernel matrices used in standard
%       diffusion-map embedding of neuroimaging co-activity data.
%
%   See also:
%       GRADIENTS, LOYVAIN.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    p (1, 1) double {mustBeInRange(p, 0, 1)} = 0.1
end

A = W > quantile(W, (1-p), 2);
B = A * A';
