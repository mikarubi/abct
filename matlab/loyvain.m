function [M, Q] = loyvain(X, k, objective, similarity, varargin)
% LOYVAIN Normalized modularity, k-means, or spectral clustering
%
%   [M, Q] = loyvain(X, k)
%   [M, Q] = loyvain(X, k, objective, similarity)
%   [M, Q] = loyvain(X, k, objective, similarity, Name=Value)
%
%   Inputs:
%       X:  Network matrix of size n x n, where
%           n is the number of nodes.
%       OR  Data matrix of size n x t, where
%           n is the number of features and
%           t is the number of observations.
%
%       k: Number of modules (positive integer or 0).
%           Set to 0 to infer number from initial module assignment.
%
%       objective: Clustering objective.
%           "modularity": Normalized modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Spectral clustering objective.
%
%       similarity: Type of similarity.
%         The first option assumes that X is a network matrix.
%           "network": Network connectivity (default).
%               X is a symmetric network matrix. The network must
%               be non-negative for the spectral and modularity
%               objectives. No additional similarity is computed.
%         The other options assume that X is a data matrix.
%           "corr": Pearson correlation coefficient.
%               A magnitude-normalized dot product of mean-centered vectors.
%           "cosim": Cosine similarity.
%               A normalized dot product.
%           "cov":  Covariance.
%               A dot product of mean-centered vectors.
%           "dot": Dot product.
%               A sum of an elementwise vector product.
%
%       Name=[Value] Arguments:
%
%           Start=[Initial module assignments].
%               "greedy": Maximin (maximally greedy kmeans++) initialization (default).
%               "balanced": Standard kmeans++ initialization.
%               "random": Uniformly random initialization.
%               Initial-module-assignment vector of length n.
%
%           NumBatches=[Number of batches].
%               Positive integer (default is 2).
%
%           MaxIter=[Maximum number of algorithm iterations].
%               Positive integer (default is 1000).
%
%           Replicates=[Number of replicates].
%               Positive integer (default is 10).
%
%           Tolerance=[Convergence tolerance].
%               Positive scalar (default is 1e-10).
%
%           Display=[Display progress].
%               "none": no display (default).
%               "replicate": display progress at each replicate.
%               "iteration": display progress at each iteration.
%
%   Outputs:
%       M: Vector of module assignments (length n).
%       Q: Value of maximized objective.
%
%   Methodological notes:
%       Loyvain is a unification of:
%           Lloyd's algorithm for k-means clustering and
%           Louvain algorithm for modularity maximization.
%
%       Note 1. The normalized modularity maximization is equivalent to
%       k-means clustering of data after degree correction. When the input
%       is a data rather than a network matrix, degree correction is
%       implemented via an approximately equivalent step of global-signal
%       regression. More generally, degree correction and global-signal
%       regression are both approximately equivalent to first-mode removal,
%       or subtraction of the rank-one approximation of the data.
%
%       Note 2. For "network" similarity, the value of the normalized
%       modularity is rescaled by the following factor:
%           (average module size) / (absolute sum of all weights)
%       This rescaling approximately aligns the value of the objective
%       function with values of the unnormalized modularity. For other
%       similarity measures the value of the objective function is not
%       rescaled, but the algorithm optimizes an equivalent objective.
%
%       Note 3. The Loyvain algorithm is not guaranteed to converge if
%       all swaps are accepted at each iteration (NumBatches = 1).
%       Therefore, it is generally a good idea to set NumBatches > 1.
%
%   See also:
%       COLOYVAIN, CCA, GRADIENTS, MODEREMOVAL.

arguments
    X (:, :) double {mustBeNonempty, mustBeReal, mustBeFinite}
    k (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
    objective (1, 1) string {mustBeMember(objective, ...
        ["modularity", "kmeans", "spectral"])} = "modularity"
    similarity (1, 1) string {mustBeMember(similarity, ...
        ["network", "corr", "cosim", "cov", "dot"])} = "network"
end
arguments (Repeating)
    varargin
end

% parse, process, and test arguments
Args = loyv.step0_args("method", "loyvain", "X", X, "k", k, ...
    "objective", objective, "similarity", similarity, varargin{:});
clear X k objective similarity
Args = loyv.step1_proc_loyvain(Args);
loyv.step2_test(Args.X, Args.W, Args.n, Args.k, Args);

%% Run algorithm

Q = - inf;
for i = 1:Args.replicates
    Args.replicate_i = i;
    if Args.start == "custom"
        M0 = Args.M0;
    else
        % initialize
        M0 = loyv.step3_init(Args.X, Args.normX, Args.Dist, Args.n, Args);
    end
    [M1, Q1] = loyv.step4_run(Args, Args.W, M0);    % run
    if (Q1 - Q) > Args.tolerance                    % test for increase
        if ismember(Args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, Q1, Q1 - Q);
        end
        Q = Q1;
        M = M1;
    end
end

end
