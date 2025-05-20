function V = gradients(W, k, weight, thr, varargin)
% GRADIENTS Low-dimensional representation of common-neighbor matrices
%
%   V = gradients(W, k)
%   V = gradients(W, k, weight)
%   V = gradients(W, k, weight, thr)
%   V = gradients(W, k, weight, thr, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       k: Number of gradient outputs.
%
%       weight: Type of gradient
%           "weighted": Weighted gradient (default).
%           "binary": Binary gradient.
%
%       thr: Threshold to define top neighbors.
%           Set thr = [] for default value.
%           See CONEIGHBORS for details.
%
%       Name=[Value] Arguments (binary gradients only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       V: Gradient matrix (size n x k).
%
%   Methodological notes:
%       Weighted gradients are eigenvectors of common-neighbors
%       matrices. These gradients are approximately equivalent to the
%       output of standard diffusion-map embedding of neuroimaging
%       co-activity data.
% 
%       Binary gradients are modules of common-neighbors matrices,
%       estimated using the Loyvain algorithm. Detection of these gradients
%       is equivalent to detection of eigenvectors of common-neighbors
%       matrices with binary constraints. Unlike eigenvectors, the order of
%       binary gradients will, in general, be arbitrary. 
%
%   See also:
%       CONEIGHBORS, LOYVAIN.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary"])} = "weighted"
    thr = []
end
arguments (Repeating)
    varargin
end

% Get common-neighbors matrix
if isempty(thr)
    B = coneighbors(W);
else
    B = coneighbors(W, thr);
end

% Get gradients
switch weight
    case "weighted"
        if ~isempty(varargin)
            warning("Ignoring Name=Value arguments for weighted gradients.")
        end
        [V, ~] = eigs(B, k+1);
        V = V(:, 2:end);
    case "binary"
        M = loyvain(B, k, "kmodularity", "network", varargin{:});
        V = full(sparse(1:length(B), M, 1));
end
