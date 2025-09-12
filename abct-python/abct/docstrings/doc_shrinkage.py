```
 SHRINKAGE Shrinkage of network or timeseries data

   X1 = shrinkage(X)

   Inputs:
       X:  Network matrix of size n x n, or data matrix of size n x p.
           n is the number of nodes or data points and
           p is the number of features.

   Outputs:
       X1: Shrunken network or timeseries matrix.

   Methodological notes:
       The shrinkage algorithm uses cubic interpolation to "despike" an
       initial eigenspectrum peak.

   See also:
       RESIDUALN.

```
