function V = gradients(W, k, type, p, varargin)
% GRADIENTS Low-dimensional representation of common-neighbor matrices
%
%   V = gradients(W, k)
%   V = gradients(W, k, type, p)
%   V = gradients(W, k, type, p, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       k: Number of gradient outputs.
%
%       type: Type of gradient
%           "weighted": Weighted gradient (default).
%           "binary": Binary gradient.
%
%       p: Fraction to define neighbors as the top-p connections.
%           Set p = [] for default value. See CONEIGHBORS for details.
%
%       Name=[Value] Arguments:
%           Name-value arguments for the Loyvain algorithm.
%           Only used if type = "binary". See LOYVAIN for details.
%
%   Outputs:
%       V: Gradient matrix (size n x k).
%
%   Methodological notes:
%       Weighted gradients are the eigenvectors of common-neighbors
%       matrices. These gradients are approximately equivalent to the
%       output of standard diffusion-map embedding of neuroimaging
%       co-activity data. Binary gradients are the modules of
%       common-neighbors matrices, estimated using the Loyvain algorithm.
%
%   See also:
%       CONEIGHBORS, LOYVAIN.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["weighted", "binary"])} = "weighted"
    p = []
end
arguments (Repeating)
    varargin
end
if type == "weighted" && ~isempty(varargin)
    warning("Ignoring Name=Value arguments for weighted gradients.")
end

% Get common-neighbors matrix
if isempty(p)
    B = coneighbors(W);
else
    B = coneighbors(W, p);
end

% Get gradients
switch type
    case "weighted"
        [V, ~] = eigs(B, k+1);
        V = V(:, 2:end);
    case "binary"
        M = loyvain(B, k, "modularity", varargin{:}, similarity="network");
        V = full(sparse(1:length(B), M, 1));
end
