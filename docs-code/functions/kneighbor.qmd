```
 KNEIGHBOR Common-neighbor or symmetric kappa-nearest-neighbor matrix

   B = kneighbor(W)
   B = kneighbor(W, type)
   B = kneighbor(W, type, kappa)
   B = kneighbor(X, type, kappa, similarity)
   B = kneighbor(X, type, kappa, similarity, method)
   B = kneighbor(X, type, kappa, similarity, method, Name=Value)

   Inputs:

       W: Network matrix of size n x n.
       OR
       X: Data matrix of size n x p, where
           n is the number of data points and
           p is the number of features.

       type: Type of neighbor matrix.
           "common": Common-neighbor matrix (default).
           "nearest": Symmetric kappa-nearest neighbor matrix.

       kappa: Number of nearest neighbors.
           1 <= kappa < n (default is 10).
           OR
           0 < kappa < 1 to use as a fraction of n.

       similarity: Type of similarity.
           "network": Network connectivity (default).
           "corr": Pearson correlation coefficient.
           "cosim": Cosine similarity.

       method: Method of neighbor search.
           "direct": Direct computation of similarity matrix (default).
           "indirect": knnsearch (in MATLAB)
                       pynndescent (in Python).

       Name=Value Arguments:
           Optional arguments passed to knnsearch or pynndescent.

   Outputs:
       B: Co-neighbor or symmetric nearest-neighbor matrix (size n x n).

   Methodological notes:
       Symmetric kappa-nearest-neighbor matrices are binary matrices that
       connect pairs of nodes if one of the nodes is a top-kappa nearest
       neighbor of the other node (in a structural, correlation, or
       another network).

       kappa-common-neighbor matrices are symmetric integer matrices that
       connect pairs of nodes by the number of their shared top-kappa
       nearest neighbors.
 
       Direct computation of the similarity matrix is performed in
       blocks. It is generally faster than indirect computation.

   Dependencies:
       MATLAB: 
           Statistics and Machine Learning Toolbox (if method="indirect")
       Python: 
           PyNNDescent (if method="indirect")

   See also:
       KNEICOMP, MUMAP.

```
