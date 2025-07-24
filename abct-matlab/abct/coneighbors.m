function B = coneighbors(W, thr)
% CONEIGHBORS Common-neighbors matrix of network matrix
%
%   B = coneighbors(W)
%   B = coneighbors(W, thr)
%
%   Inputs:
%
%       W: Network matrix of size n x n.
%
%       thr: Threshold to define top neighbors.
%           0 < thr < 1 (default is 0.1).
%
%   Outputs:
%       B: Common-neighbors matrix (size n x n).
%
%   Methodological notes:
%       Integer-valued common-neighbors matrices denote the number of
%       common top-l neighbors between all pairs of nodes. These matrices
%       are approximately equivalent to the kernel matrices used in
%       standard diffusion-map embedding of neuroimaging co-activity data.
%
%   See also:
%       GRADIENTS.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    thr (1, 1) double {mustBeInRange(thr, 0, 1)} = 0.1
end

n = length(W);
W(1:n+1:end) = -inf;
A = W > quantile(W, (1-thr), 2);
B = A * A';
