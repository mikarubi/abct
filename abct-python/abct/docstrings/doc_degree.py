```
 DEGREE Degree of network matrix

   S = degree(W)
   S = degree(W, type)

   Inputs:
       W: Network matrix of size n x n.

       type: Degree type
           "first": (First) degree (default).
           "second": Second degree.
           "residual": Degree after global residualization.

   Outputs:
       S: Degree vector (length n).

   Methodological notes:
       The first degree is the sum of connection weights. The second
       degree is the sum of squared connection weights. Together, the
       first and second degrees are exactly or approximately equivalent to
       several measures of network communication and control.

       The residual degree is the degree after first-component removal and
       can be approximately equivalent to the primary co-activity gradient
       in functional MRI co-activity networks.

   See also:
       RESIDUALN, DISPERSION.

```
