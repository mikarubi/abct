function [M, Q] = loyvain(varargin)
% LOYVAIN K-modularity, k-means, or spectral clustering
%
%   [M, Q] = loyvain(W, k)
%   [M, Q] = loyvain(X, k)
%   [M, Q] = loyvain(_, objective, similarity)
%   [M, Q] = loyvain(_, objective, similarity, Name=Value)
%
%   Inputs:
%       W:  Network matrix of size n x n.
%
%       OR
%
%       X:  Data matrix of size n x t, where
%           n is the number of data points and
%           t is the number of features.
%
%       k: Number of modules (positive integer or 0).
%           Set to 0 to infer number from initial module assignment.
%
%       objective: Clustering objective.
%           "kmodularity": K-modularity (default).
%           "kmeans": K-means clustering objective.
%           "spectral": Spectral clustering objective (normalized cut).
%
%       similarity: Type of similarity.
%         The first option assumes that the first input is W, a network matrix.
%           "network": Network connectivity (default).
%               W is a symmetric network matrix. The network must
%               be non-negative for k-modularity and spectral
%               objectives. No additional similarity is computed.
%         The other options assume that the first input is X, a data matrix.
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
%       Lloyd's algorithm for k-means clustering and
%       Louvain algorithm for modularity maximization.
%
%       K-modularity maximization is exactly equivalent to normalized
%       modularity maximization and approximately equivalent to k-means
%       clustering after first-mode removal.
%       * When the input is a network matrix, first-mode
%         removal is implemented via degree correction.
%       * When the input is a data matrix, first-mode removal
%         is implemented via global-signal regression.
%
%       For "network" similarity, k-modularity is rescaled by:
%           (average module size) / (absolute sum of all weights)
%       This rescaling aligns k-modularity within the range of the
%       modularity, but has no effect on the optimization algorithm.
%
%       The Loyvain algorithm is not guaranteed to converge if
%       all swaps are accepted at each iteration (NumBatches = 1).
%       Therefore, it is generally a good idea to set NumBatches > 1.
%
%   See also:
%       COLOYVAIN, CANONCOV, GRADIENTS, MODEREMOVAL.

%% Parse, process, and test arguments

% Parse arguments
Args = loyv.step0_args("loyvain", varargin{:});
clear varargin

% Process all other arguments
Args = loyv.step1_proc_loyvain(Args);

% Test arguments
loyv.step2_test(Args.X, Args.W, Args.n, Args.k, Args);

%% Run algorithm

Q = -inf;
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
