function B = kneighbor(W, type, kappa, similarity, method, varargin)
% KNEIGHBOR Common-neighbor or symmetric kappa-nearest-neighbor matrix
%
%   B = kneighbor(W)
%   B = kneighbor(W, type)
%   B = kneighbor(W, type, kappa)
%   B = kneighbor(X, type, kappa, similarity)
%   B = kneighbor(X, type, kappa, similarity, method)
%   B = kneighbor(X, type, kappa, similarity, method, Name=Value)
%
%   Inputs:
%
%       W: Network matrix of size n x n.
%       OR
%       X: Data matrix of size n x p, where
%           n is the number of data points and
%           p is the number of features.
%
%       type: Type of neighbor matrix.
%           "common": Common-neighbor matrix (default).
%           "nearest": Symmetric kappa-nearest neighbor matrix.
%
%       kappa: Number of nearest neighbors.
%           1 <= kappa < n (default is 10).
%           OR
%           0 < kappa < 1 to use as a fraction of n.
%
%       similarity: Type of similarity.
%           "network": Network connectivity (default).
%           "corr": Pearson correlation coefficient.
%           "cosim": Cosine similarity.
%
%       method: Method of neighbor search.
%           "direct": Direct computation of similarity matrix (default).
%           "indirect": knnsearch (in MATLAB)
%                       pynndescent (in Python).
%
%       Name=[Value] Arguments:
%           Optional arguments passed to knnsearch or pynndescent.
%
%   Outputs:
%       B: Co-neighbor or symmetric nearest-neighbor matrix (size n x n).
%
%   Methodological notes:
%       Symmetric kappa-nearest-neighbor matrices are binary matrices that
%       connect pairs of nodes if one of the nodes is a top-kappa nearest
%       neighbor of the other node (in a structural, correlation, or
%       another network).
%
%       kappa-common-neighbor matrices are symmetric integer matrices that
%       connect pairs of nodes by the number of their shared top-kappa
%       nearest neighbors.
% 
%       Direct computation of the similarity matrix is performed in
%       blocks. It is generally faster than indirect computation.
%
%   Dependencies:
%       MATLAB: 
%           Statistics and Machine Learning Toolbox (if method="indirect")
%       Python: 
%           PyNNDescent (if method="indirect")
%
%   See also:
%       KNEICOMP, MUMAP.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    type (1, 1) string {mustBeMember(type, ["common", "nearest"])} = "common"
    kappa (1, 1) double {mustBePositive} = 10
    similarity (1, 1) string {mustBeMember(similarity, ["network", "corr", "cosim"])} = "network"
    method (1, 1) string {mustBeMember(method, ["direct", "indirect"])} = "direct"
end
arguments (Repeating)
    varargin
end

n = size(W, 1);

if kappa < 1
    kappa = clip(round(n * kappa), 1, n-1);
else
    assert(kappa < n, "kappa must be less than number of nodes or data points.")
    assert(isequal(kappa, round(kappa)), "kappa > 1 must be an integer.")
end

Row = repmat((1:n)', 1, kappa+1);
if similarity == "network"
    assert(isequal(size(W, 1), size(W, 2)), "Network matrix must be square.")
    W(1:n+1:end) = inf;
    [~, Col] = maxk(W, kappa+1, 2);
else
    X = W;
    switch method
        case "direct"
            % Center to mean 0 for correlation
            if similarity == "corr"
                X = X - mean(X, 2);
            end
            % Rescale to norm 1
            X = X ./ vecnorm(X, 2, 2);

            % Compute similarity matrix in blocks of 1e8 elements
            % It follows that n * nb = 1e8, b = n / nb = n^2 / 1e8
            b = ceil(n^2 / 1e8);
            b = clip(b, 1, n);
            Ix = floor(linspace(1, n+1, b+1));

            Col = zeros(n, kappa+1);
            for i = 1:b
                Ixi = Ix(i):Ix(i+1)-1;
                [~, Col(Ixi, :)] = maxk(X(Ixi, :) * X', kappa+1, 2);
            end
        case "indirect"
            switch similarity
                case "corr"; knnsim = "correlation";
                case "cosim"; knnsim = "cosine";
            end
            Col = knnsearch(X, X, "K", kappa+1, "Distance", knnsim, varargin{:});
    end
end
A = sparse(Row(:), Col(:), 1, n, n);
A(1:n+1:end) = 0;

switch type
    case "common"
        B = A * A';
    case "nearest"
        B = double(A | A');
end
