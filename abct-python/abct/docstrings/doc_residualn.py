```
 RESIDUALN Global residualization of network or timeseries data

   X1 = residualn(X)
   X1 = residualn(X, type)

   Inputs:
       X:  Network matrix of size n x n, or data matrix of size n x p.
           n is the number of nodes or data points and
           p is the number of features.

       type: Type of global residualization.
           "degree": Degree correction (default).
           "global": Global signal regression.
           "rankone": Subtraction of rank-one approximation.

   Outputs:
       X1: Residual network or timeseries matrix.

   See also:
       SHRINKAGE.

```
