```
 LOYVAIN K-modularity, k-means, or spectral clustering

   M, Q = loyvain(W, k)
   M, Q = loyvain(W, k, objective)
   M, Q = loyvain(X, k, objective, similarity)
   M, Q = loyvain(_, objective, similarity, Name=Value)

   Inputs:
       W:  Network matrix of size n x n.
       OR
       X:  Data matrix of size n x p, where
           n is the number of data points and
           p is the number of features.

       k: Number of modules (positive integer or 0).
           Set to 0 to infer number from initial module assignment.

       objective: Clustering objective.
           "kmodularity": K-modularity (default).
           "kmeans": K-means clustering objective.
           "spectral": Spectral clustering objective (normalized cut).

       similarity: Type of similarity.
         The first option assumes that the first input is a network matrix.
           "network": Network connectivity (default).
               W is a symmetric network matrix. The network must
               be non-negative for k-modularity and spectral
               objectives. No additional similarity is computed.
         The other options assume that the first input is X, a data matrix.
           "corr": Pearson correlation coefficient.
               A magnitude-normalized dot product of mean-centered vectors.
           "cosim": Cosine similarity.
               A normalized dot product.
           "cov":  Covariance.
               A dot product of mean-centered vectors.
           "dot": Dot product.
               A sum of an elementwise vector product.

       Name=Value Arguments:

           start=Initial module assignments.
               "greedy": Maximin (greedy kmeans++) initialization (default).
               "balanced": Standard kmeans++ initialization.
               "random": Uniformly random initialization.
               Numeric vector: Initial module assignment vector of length n.

           numbatches=Number of batches.
               Positive integer (default is 10).

           maxiter=Maximum number of algorithm iterations.
               Positive integer (default is 1000).

           replicates=Number of replicates.
               Positive integer (default is 10).

           tolerance=Convergence tolerance.
               Positive scalar (default is 1e-10).

           display=Display progress.
               "none": no display (default).
               "replicate": display progress at each replicate.
               "iteration": display progress at each iteration.

   Outputs:
       M: Vector of module assignments (length n).
       Q: Value of maximized objective.

   Methodological notes:
       Loyvain is a unification of:
       Lloyd's algorithm for k-means clustering and
       Louvain algorithm for modularity maximization.

       K-modularity maximization is exactly equivalent to normalized
       modularity maximization and approximately equivalent to k-means
       clustering with global residualization. Global residualization is
       implemented as degree correction for network matrices and
       global-signal regression for data matrices.

       For "network" similarity, k-modularity is rescaled by:
           (average module size) / (absolute sum of all weights)
       This rescaling aligns k-modularity within the range of the
       modularity, but has no effect on the optimization algorithm.

       The Loyvain algorithm is not guaranteed to converge if
       all swaps are accepted at each iteration (NumBatches = 1).
       Therefore, it is generally a good idea to set NumBatches > 1.

   See also:
       COLOYVAIN, CANONCOV, KNEICOMP, RESIDUALN.

```
