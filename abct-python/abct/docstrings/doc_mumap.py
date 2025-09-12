```
 MUMAP m-umap low-dimensional embedding

   U = mumap(X);
   U, Partition, CostHistory = mumap(X, Name=Value);

   Inputs:
       W:  Network matrix of size n x n.

       OR

       X:  Data matrix of size n x p, where
           n is the number of data points and
           p is the number of features.

       Name=Value Arguments:
           d=Embedding dimension.
               Positive integer (default is 3).

           kappa=Number of nearest neighbors.
               Positive integer (default is 30).

           alpha=Inverse amplitude of long-range attraction.
               Positive scalar >= 1 (default is 1).
               Larger alpha implies weaker long-range attraction.

           beta=Slope of long-range attraction.
               Positive scalar <= 1 (default is 1).
               Larger beta implies faster decay of attraction.

           gamma=Modularity resolution parameter.
               Positive scalar (default is 1).

           similarity=Type of similarity.
               "network": Network connectivity (default).
               "corr": Pearson correlation coefficient.
               "cosim": Cosine similarity.

           method=Method of nearest-neighbor search.
               "direct": Direct computation of similarity matrix.
               "indirect": knnsearch (MATLAB) or pynndescent (Python).

           replicates=Number of modularity replicates.
               Positive integer (default is 10).

           finaltune=Modularity final tuning.
               Logical scalar (default is true).

           partition=Module partition.
               Integer vector: module partition of length n (default is ).

           start=Initial embedding method.
               "greedy": Spherical maximin initialization (default).
               "spectral": Spectral initialization of module (k x k) matrix.
               "spectral_nn": Spectral initialization of full (n x n) matrix.
               Numeric matrix: Initial embedding of size n x d, where:
                   n is the number of data points and
                   d is the embedding dimension.

           solver=Optimization solver.
               "trustregions": Manopt trust-regions method (default for MATLAB).
               "adam": Adaptive Moment Estimation optimizer (default for Python).

           maxiter=Maximum number of iterations.
               Positive integer (default is 10000).

           learnrate=Optimizer learning rate.
               Positive scalar (default is 0.001).

           tolerance=Solution tolerance.
               Positive scalar (default is 1e-6).

           gpu=Use GPU.
               Logical (default is false).

           cache=Cache gradient matrices.
               Logical (default is false).

           verbose=Verbose output.
               Logical (default is true).

   Outputs:
       U:  Embedding matrix of size n x d.
       Partition: Module partition of length n.
       CostHistory: Cost history of optimization.

   Methodological notes:
       m-umap is a first-order approximation of the true parametric
       loss of UMAP, with spherical constraints. It is simultaneously
       equivalent to the modularity, and to spring layout methods with
       Cauchy components and spherical constraints.

   Dependencies:
       MATLAB: 
           Statistics and Machine Learning Toolbox (if method="indirect")
           Deep Learning Toolbox (if solver="adam")
           Manopt (if solver="trustregions")
       Python: 
           igraph
           PyTorch
           PyNNDescent (if method="indirect")
           PyManopt (if solver="trustregions") * warning: slow *

```
