```
 CANONCOV Canonical covariance analysis (aka partial least squares)
          Canonical correlation analysis

   A, B, U, V, R = canoncov(X, Y, k)
   A, B, U, V, R = canoncov(X, Y, k, type)
   A, B, U, V, R = canoncov(X, Y, k, type, resid)
   A, B, U, V, R = canoncov(X, Y, k, type, resid, corr)
   A, B, U, V, R = canoncov(X, Y, k, type, resid, corr, Name=Value)

   Inputs:
       X: Data matrix of size n x p, where
          n is the number of data points and
          p is the number of features.

       Y: Data matrix of size n x q, where
          n is the number of data points and
          q is the number of features.

       k: Number of canonical components (positive integer).

       type: Weighted or binary canonical analysis.
           "weighted": Weighted canonical analysis (default).
           "binary": Binary canonical analysis.

       resid: Global residualization (logical scalar).
           0: No global residualization.
           1: Global residualization via degree correction (default).

       corr: Canonical correlation analysis (logical scalar).
           0: Canonical covariance analysis (default).
           1: Canonical correlation analysis.

       Name=Value Arguments
           (binary canonical analysis only):
           See LOYVAIN for all Name=Value options.

   Outputs:
       A: Canonical coefficients of X (size p x k).
       B: Canonical coefficients of Y (size q x k).
       U: Canonical components of X (size n x k).
       V: Canonical components of Y (size n x k).
       R: Canonical covariances or correlations (size k x k).
          If type is "weighted", R denotes the actual covariances or
          correlations. If type is "binary", R denotes the
          normalized covariances or correlations.

   Methodological notes:
       Weighted canonical correlation or covariance analysis is computed
       via singular value decomposition of cross-covariance matrix.

       Binary canonical covariance analysis is computed via co-Loyvain
       k-means clustering of cross-covariance matrix. This analysis
       produces binary orthogonal canonical coefficients.

       Binary canonical covariance analysis is computed via co-Loyvain
       k-means clustering of _whitened_ cross-covariance matrix. This
       analysis produces binary orthogonal canonical coefficients for
       the whitened matrix. However, the output coefficients after
       dewhitening will, in general, not be binary.

       Global residualization is implemented via generalized degree
       correction, and converts k-means co-clustering into k-modularity
       co-maximization.

   See also:
       COLOYVAIN, LOYVAIN, RESIDUALN.

```
