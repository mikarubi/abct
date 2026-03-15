```
 RESIDUALN Residualization of network or data matrix

   W1 = residualn(W)
   W1 = residualn(W, type)
   X1 = residualn(X)
   X1 = residualn(X, type)

   Inputs:
       W:  Network matrix of size n x n.
       OR
       X:  Data matrix of size n x p, where
           n is the number of data points and
           p is the number of features.

       type: Type of residualization.
           "degree": Degree correction (default)
               Subtraction of the rescaled product of the degrees.
           "degree_ctr": Double centering
               Subtraction of the rescaled and shifted degrees.
           "global": Global signal regression
               Regression out of the global signal (column mean).
           "global_ctr": Global signal subtraction (centering)
               Subtraction of the global signal (column mean).
           "rankone": Rank-one subtraction
               Subtraction of the rank-one approximation.

   Outputs:
       W1: Residual network matrix.
       OR
       X1: Residual data matrix.

   See also:
       SHRINKAGE.

```
