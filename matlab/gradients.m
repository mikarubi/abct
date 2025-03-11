function V = gradients(W, k, p, args)
% GRADIENTS Low-dimensional representation of common-neighbor matrices
%
%   V = gradients(W, k)
%   V = gradients(W, k, p, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%       k: Number of gradient outputs.
%       p: Neighbor fraction
%           Neighbors are the top-p fraction of connections
%           0 < p < 1 (default is 0.1).
%
%       Name=[Value] Arguments (Optional):
%
%           Type=[Type of common neighbors].
%               "weighted": Eigenvectors of common-neighbors matrix (default).
%               "binary": Modules of common neighbors.
%
%   Outputs:
%       V: Eigenvectors or modules of common-neighbors matrix.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    p (1, 1) double {mustBeInRange(p, 0, 1)} = 0.1
    args.Type (1, 1) string {mustBeMember(args.Type, ["weighted", "binary"])} = "weighted"
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
