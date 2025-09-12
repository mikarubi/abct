```
 DISPERSION Dispersion of network matrix

   D = dispersion(W)
   D = dispersion(W, type, M)

   Inputs:
       W: Network matrix of size n x n.

       type: Dispersion type
           "coefvar2": Squared coefficient of variation (default).
           "kpartcoef": k-Participation coefficient.

       M: Module vector of length n (if type is "kpartcoef" only).

   Outputs:
       D: Dispersion vector (length n).

   Methodological notes:
       The squared coefficient of variation, or CV2, is the ratio of the
       variance to the square of the mean. CV2 is equivalent to the ratio
       of the second moment to the square of the first moment.

       The participation coefficient is a popular module-based measure of
       connectional diversity. The k-participation coefficient is the
       participation coefficient normalized by module size.

       CV2 is approximately equivalent to the k-participation coefficient
       in homogeneously modular networks, such as correlation or
       co-neighbor networks.

   See also:
       DEGREE.

```
