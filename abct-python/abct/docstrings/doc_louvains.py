```
 LOUVAINS Efficient Louvain modularity maximization of sparse networks (MATLAB)
 LEIDEN igraph Leiden modularity maximization (Python)

   M, Q = louvains(W, Name=Value)        % MATLAB
   M, Q = leiden(W, Name=Value)          # Python

   Inputs:
       W:  Network matrix of size n x n.

       Name=Value Arguments:

           gamma=Resolution parameter.
               Positive scalar (default is 1).

           start=Initial module assignments.
               Vector of length n (default is 1:n).

           replicates=Number of replicates.
               Positive integer (default is 10).

           finaltune=Final tuning of optimized assignment.
               Logical (default is false).

           tolerance=Convergence tolerance.
               Positive scalar (default is 1e-10).

           display=Display progress.
               "none": no display (default).
               "replicate": display progress at each replicate.

   Outputs:
       M: Vector of module assignments (length n).
       Q: Value of maximized modularity.

   See also:
       MUMAP, LOYVAIN.

```
