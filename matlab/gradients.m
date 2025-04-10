function V = gradients(W, k, weight, p, varargin)
% GRADIENTS Low-dimensional representation of common-neighbor matrices
%
%   V = gradients(W, k)
%   V = gradients(W, k, weight)
%   V = gradients(W, k, weight, p)
%   V = gradients(W, k, weight, p, Name=Value)
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
%       p: Fraction to define neighbors as the top-p connections.
%           Set p = [] for default value. See CONEIGHBORS for details.
%
%       Name=[Value] Arguments (binary gradients only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       V: Gradient matrix (size n x k).
%
%   Methodological notes:
%       Weighted gradients are the eigenvectors of common-neighbors
%       matrices. These gradients are approximately equivalent to the
%       output of standard diffusion-map embedding of neuroimaging
%       co-activity data. Binary gradients are the modules of common-
%       neighbors matrices, estimated using the Loyvain algorithm. Note
%       that the order of binary gradients is generally arbitrary. 
%
%   See also:
%       CONEIGHBORS, LOYVAIN.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary"])} = "weighted"
    p = []
end
arguments (Repeating)
    varargin
end

% Get common-neighbors matrix
if isempty(p)
    B = coneighbors(W);
else
    B = coneighbors(W, p);
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
