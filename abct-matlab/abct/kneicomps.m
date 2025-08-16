function V = kneicomps(W, k, weight, thr, varargin)
% KNEICOMPS Components of neighbor matrices
%
%   V = kneicomps(W, k)
%   V = kneicomps(W, k, weight)
%   V = kneicomps(W, k, weight, thr)
%   V = kneicomps(W, k, weight, thr, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       k: Number of components.
%
%       weight: Type of components
%           "weighted": Weighted components (default).
%           "binary": Binary components.
%
%       thr: Threshold to define top neighbors.
%           Set thr = [] for default value.
%           See KNEIGHBORS for details.
%
%       Name=[Value] Arguments (binary components only):
%           See LOYVAIN for all Name=Value options.
%
%   Outputs:
%       V: Component matrix (size n x k).
%
%   Methodological notes:
%       Weighted components are eigenvectors of common-neighbors
%       matrices. In imaging neuroscience, these components are
%       approximately equivalent to co-activity gradients 
%       (diffusion-map embeddings).
% 
%       Binary components are modules of common-neighbors matrices,
%       estimated using the Loyvain algorithm. They are equivalent to
%       eigenvectors of common-neighbors matrices with binary constraints.
%       The order of binary components will, in general, be arbitrary.
%
%   See also:
%       KNEIGHBORS, LOYVAIN.

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
    B = kneighbors(W);
else
    B = kneighbors(W, thr);
end

% Get components
switch weight
    case "weighted"
        if ~isempty(varargin)
            warning("Ignoring Name=Value arguments for weighted components.")
        end
        [V, ~] = eigs(B, k+1);
        V = V(:, 2:end);
    case "binary"
        M = loyvain(B, k, "kmodularity", "network", varargin{:});
        V = full(sparse(1:length(B), M, 1));
end
