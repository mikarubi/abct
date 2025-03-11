function V = gradients(W, k, type, p, varargin)
% GRADIENTS Low-dimensional representation of common-neighbor matrices
%
%   V = gradients(W, k)
%   V = gradients(W, k, p, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%       k: Number of gradient outputs.
%
%   Optional Inputs:
%       type: Type of gradient
%           "weighted": Weighted gradient (default).
%           "binary": Binary gradient.
%
%       p: Fraction to define neighbors (see CONEIGHBORS for details).
%
%       Name=[Value] Arguments.
%           Loyvain algorithm options (See LOYVAIN for details).
%
%   Outputs:
%       V: Gradient matrix of size n x k.
%
%   Methodological notes:
%       Weighted gradients are the eigenvectors of common-neighbors matrices.
%       Binary gradients are the modules of common-neighbors matrices, estimated
%       using Loyvain. These matrices are approximately equivalent to the output
%       of standard diffusion-map embedding of neuroimaging co-activity data.
%
%   See also:
%       CONEIGHBORS, LOYVAIN.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    type (1, 1) string {mustBeMember(type, ["weighted", "binary"])} = "weighted"
    p (1, 1) double {mustBeInRange(p, 0, 1)}
end
arguments (Repeating)
    varargin
end

B = coneighbors(W, p);

switch args.Type
    case "weighted"
        [V, ~] = eigs(B, k+1);
        V = V(:, 2:end);
    case "binary"
        n = length(B);
        M = loyvain(B, k, objective="modularity");
        V = full(sparse(1:n, M, 1));
end
