function V = kneicomp(W, k, weight, varargin)
% KNEICOMP Components of neighbor matrices
%
%   V = kneicomp(W, k)
%   V = kneicomp(X, k)
%   V = kneicomp(_, k, weight)
%   V = kneicomp(_, k, weight, Name=Value)
%
%   Inputs:
%       W: Network matrix of size n x n.
%       OR
%       X: Data matrix of size n x p, where
%           n is the number of data points and
%           p is the number of features.
%
%       k: Number of components.
%
%       weight: Type of components
%           "weighted": Weighted components (default).
%           "binary": Binary components.
%
%       Name=[Value] Arguments:
%           KNEIGHBOR: type, kappa, similarity, method
%               (see KNEIGHBOR for details).
%           LOYVAIN: All Name=Value arguments
%               (binary components only, see LOYVAIN for details).
%
%   Outputs:
%       V: Component matrix (size n x k).
%
%   Methodological notes:
%       By default, weighted components are eigenvectors of
%       common-neighbors matrices. In imaging neuroscience, these
%       components are approximately equivalent to co-activity gradients
%       (diffusion-map embeddings).
% 
%       Correspondingly, binary components are modules of common-neighbors 
%       matrices, estimated using the Loyvain algorithm. They are
%       equivalent to eigenvectors of common-neighbors matrices with binary
%       constraints. The order of binary components will be arbitrary. 
%
%   See also:
%       KNEIGHBOR, LOYVAIN, MUMAP.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    k (1, 1) double {mustBeInteger, mustBePositive}
    weight (1, 1) string {mustBeMember(weight, ["weighted", "binary"])} = "weighted"
end
arguments (Repeating)
    varargin
end

% structured arguments
Args = struct(varargin{:});

% Default kneighbor arguments
args = struct(type="common", kappa=0.1, similarity="network", method="direct");
% Update arguments with Args
for name = reshape(string(fieldnames(args)), 1, [])
    if isfield(Args, name)
        args.(name) = Args.(name);
        Args = rmfield(Args, name);
    end
end
varargin = namedargs2cell(Args);

% Get a k-neighbors matrix
B = kneighbor(W, args.type, args.kappa, args.similarity, args.method);

% Get components
switch weight
    case "weighted"
        if ~isempty(varargin)
            warning("Ignoring Name=Value arguments for weighted components.")
        end
        [V, ~] = eigs(B, k+1);
        V = V(:, 2:end);
    case "binary"
        M = loyvain(full(B), k, "kmodularity", "network", varargin{:});
        V = full(sparse(1:length(B), M, 1));
end
